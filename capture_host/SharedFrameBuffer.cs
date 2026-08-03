using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Threading;

namespace MapleCaptureHost;

internal sealed class SharedFrameBuffer : IDisposable
{
    public const string MappingName = "Local\\AutoMaple.GraphicsCapture.Frame";
    public const string ReadyEventName = "Local\\AutoMaple.GraphicsCapture.FrameReady";
    public const uint Magic = 0x50414D41; // AMAP
    public const int Version = 1;
    public const int HeaderSize = 64;
    public const int BytesPerPixel = 4;

    private readonly MemoryMappedFile _mapping;
    private readonly MemoryMappedViewAccessor _view;
    private readonly EventWaitHandle _frameReady;
    private readonly int _capacity;
    private long _frameId;

    public SharedFrameBuffer(int maxWidth = 2560, int maxHeight = 1440)
    {
        if (maxWidth <= 0 || maxHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxWidth));
        }

        _capacity = checked(HeaderSize + maxWidth * maxHeight * BytesPerPixel);
        _mapping = MemoryMappedFile.CreateOrOpen(MappingName, _capacity, MemoryMappedFileAccess.ReadWrite);
        _view = _mapping.CreateViewAccessor(0, _capacity, MemoryMappedFileAccess.ReadWrite);
        _frameReady = new EventWaitHandle(false, EventResetMode.AutoReset, ReadyEventName);
        WriteStatus(0, 0, 0, 0, 0, 0);
    }

    public void Publish(ReadOnlySpan<byte> bgra, int width, int height, int stride, long timestampTicks)
    {
        if (width <= 0 || height <= 0 || stride < width * BytesPerPixel)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }

        var payloadSize = checked(stride * height);
        if (HeaderSize + payloadSize > _capacity)
        {
            throw new InvalidOperationException($"Frame {width}x{height} stride={stride} exceeds shared buffer capacity.");
        }
        if (bgra.Length < payloadSize)
        {
            throw new ArgumentException("Source frame is smaller than the declared payload.", nameof(bgra));
        }

        // Odd sequence means a writer is in progress. Even sequence means stable data.
        var nextFrameId = Interlocked.Increment(ref _frameId);
        var writeSequence = checked(nextFrameId * 2 - 1);
        _view.Write(8, writeSequence);
        _view.WriteArray(HeaderSize, bgra[..payloadSize].ToArray(), 0, payloadSize);
        WriteStatus(width, height, stride, payloadSize, timestampTicks, checked(nextFrameId * 2));
        _view.Flush();
        _frameReady.Set();
    }

    public void PublishStopped()
    {
        WriteStatus(0, 0, 0, 0, DateTime.UtcNow.Ticks, checked(Interlocked.Read(ref _frameId) * 2));
        _view.Flush();
        _frameReady.Set();
    }

    private void WriteStatus(int width, int height, int stride, int payloadSize, long timestampTicks, long sequence)
    {
        _view.Write(0, Magic);
        _view.Write(4, Version);
        _view.Write(8, sequence);
        _view.Write(16, width);
        _view.Write(20, height);
        _view.Write(24, stride);
        _view.Write(28, payloadSize);
        _view.Write(32, timestampTicks);
        _view.Write(40, Environment.ProcessId);
        _view.Write(44, HeaderSize);
    }

    public void Dispose()
    {
        PublishStopped();
        _frameReady.Dispose();
        _view.Dispose();
        _mapping.Dispose();
    }
}
