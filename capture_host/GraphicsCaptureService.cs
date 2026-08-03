using System.Runtime.InteropServices;
using Windows.Foundation;
using Windows.Graphics;
using Windows.Graphics.Capture;
using Windows.Graphics.DirectX;
using Windows.Graphics.DirectX.Direct3D11;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;
using WinRT;
using static Vortice.Direct3D11.D3D11;

namespace MapleCaptureHost;

internal sealed class GraphicsCaptureService : IDisposable
{
    private readonly SharedFrameBuffer _sharedBuffer;
    private readonly ID3D11Device _device;
    private readonly ID3D11DeviceContext _context;
    private readonly IDirect3DDevice _winRtDevice;
    private readonly GraphicsCaptureItem _item;
    private Direct3D11CaptureFramePool? _framePool;
    private GraphicsCaptureSession? _session;
    private ID3D11Texture2D? _staging;
    private SizeInt32 _size;
    private int _processing;
    private bool _disposed;

    public GraphicsCaptureService(IntPtr hwnd, SharedFrameBuffer sharedBuffer)
    {
        _sharedBuffer = sharedBuffer;
        _item = CaptureInterop.CreateItemForWindow(hwnd);
        _size = _item.Size;

        FeatureLevel[] featureLevels =
        {
            FeatureLevel.Level_11_1,
            FeatureLevel.Level_11_0,
            FeatureLevel.Level_10_1,
            FeatureLevel.Level_10_0
        };

        ID3D11Device? device;
        ID3D11DeviceContext? context;
        D3D11CreateDevice(
            adapter: null,
            driverType: DriverType.Hardware,
            flags: DeviceCreationFlags.BgraSupport,
            featureLevels: featureLevels,
            device: out device,
            immediateContext: out context).CheckError();

        _device = device ?? throw new InvalidOperationException("D3D11CreateDevice returned no device.");
        _context = context ?? throw new InvalidOperationException("D3D11CreateDevice returned no immediate context.");
        _winRtDevice = CaptureInterop.CreateWinRtDevice(_device);
    }

    public void Start()
    {
        ThrowIfDisposed();
        if (_session is not null)
        {
            return;
        }

        _framePool = Direct3D11CaptureFramePool.CreateFreeThreaded(
            _winRtDevice,
            DirectXPixelFormat.B8G8R8A8UIntNormalized,
            2,
            _size);
        _framePool.FrameArrived += OnFrameArrived;

        _session = _framePool.CreateCaptureSession(_item);
        _session.IsCursorCaptureEnabled = false;
        _session.StartCapture();
    }

    private void OnFrameArrived(Direct3D11CaptureFramePool sender, object args)
    {
        if (Interlocked.Exchange(ref _processing, 1) != 0)
        {
            return;
        }

        try
        {
            using var frame = sender.TryGetNextFrame();
            if (frame is null)
            {
                return;
            }

            var contentSize = frame.ContentSize;
            if (contentSize.Width <= 0 || contentSize.Height <= 0)
            {
                return;
            }

            if (contentSize.Width != _size.Width || contentSize.Height != _size.Height)
            {
                _size = contentSize;
                sender.Recreate(
                    _winRtDevice,
                    DirectXPixelFormat.B8G8R8A8UIntNormalized,
                    2,
                    _size);
                RecreateStaging(_size.Width, _size.Height);
                return;
            }

            using var source = CaptureInterop.GetTexture(frame.Surface);
            EnsureStaging(contentSize.Width, contentSize.Height);
            _context.CopyResource(_staging!, source);

            var mapped = _context.Map(
                _staging!,
                0,
                MapMode.Read,
                Vortice.Direct3D11.MapFlags.None);
            try
            {
                var rowBytes = checked(contentSize.Width * 4);
                var packed = GC.AllocateUninitializedArray<byte>(checked(rowBytes * contentSize.Height));
                unsafe
                {
                    var srcBase = (byte*)mapped.DataPointer;
                    fixed (byte* dstBase = packed)
                    {
                        for (var y = 0; y < contentSize.Height; y++)
                        {
                            Buffer.MemoryCopy(
                                srcBase + y * mapped.RowPitch,
                                dstBase + y * rowBytes,
                                rowBytes,
                                rowBytes);
                        }
                    }
                }

                _sharedBuffer.Publish(
                    packed,
                    contentSize.Width,
                    contentSize.Height,
                    rowBytes,
                    DateTime.UtcNow.Ticks);
            }
            finally
            {
                _context.Unmap(_staging!, 0);
            }
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"Capture frame failed: {exception.Message}");
        }
        finally
        {
            Volatile.Write(ref _processing, 0);
        }
    }

    private void EnsureStaging(int width, int height)
    {
        if (_staging is not null)
        {
            var description = _staging.Description;
            if (description.Width == (uint)width && description.Height == (uint)height)
            {
                return;
            }
        }

        RecreateStaging(width, height);
    }

    private void RecreateStaging(int width, int height)
    {
        _staging?.Dispose();
        _staging = _device.CreateTexture2D(new Texture2DDescription
        {
            Width = checked((uint)width),
            Height = checked((uint)height),
            MipLevels = 1,
            ArraySize = 1,
            Format = Format.B8G8R8A8_UNorm,
            SampleDescription = new SampleDescription(1, 0),
            Usage = ResourceUsage.Staging,
            BindFlags = BindFlags.None,
            CPUAccessFlags = CpuAccessFlags.Read,
            MiscFlags = ResourceOptionFlags.None
        });
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;

        if (_framePool is not null)
        {
            _framePool.FrameArrived -= OnFrameArrived;
        }
        _session?.Dispose();
        _framePool?.Dispose();
        _staging?.Dispose();
        _winRtDevice.Dispose();
        _context.Dispose();
        _device.Dispose();
    }
}

internal static class CaptureInterop
{
    private const string GraphicsCaptureItemRuntimeClass = "Windows.Graphics.Capture.GraphicsCaptureItem";

    private static readonly Guid GraphicsCaptureItemGuid = new("79C3F95B-31F7-4EC2-A464-632EF5D30760");
    private static readonly Guid GraphicsCaptureItemInteropGuid = new("3628E81B-3CAC-4C60-B7F4-23CE0E0C3356");
    private static readonly Guid D3D11Texture2DGuid = typeof(ID3D11Texture2D).GUID;

    public static GraphicsCaptureItem CreateItemForWindow(IntPtr hwnd)
    {
        IntPtr className = IntPtr.Zero;
        IntPtr interopPointer = IntPtr.Zero;
        IntPtr itemPointer = IntPtr.Zero;

        WindowsCreateString(
            GraphicsCaptureItemRuntimeClass,
            (uint)GraphicsCaptureItemRuntimeClass.Length,
            out className).ThrowOnFailure();

        try
        {
            var interopGuid = GraphicsCaptureItemInteropGuid;
            RoGetActivationFactory(className, ref interopGuid, out interopPointer).ThrowOnFailure();

            var interop = (IGraphicsCaptureItemInterop)Marshal.GetObjectForIUnknown(interopPointer);
            var itemGuid = GraphicsCaptureItemGuid;
            interop.CreateForWindow(hwnd, ref itemGuid, out itemPointer).ThrowOnFailure();

            return MarshalInterface<GraphicsCaptureItem>.FromAbi(itemPointer);
        }
        finally
        {
            if (itemPointer != IntPtr.Zero)
            {
                Marshal.Release(itemPointer);
            }
            if (interopPointer != IntPtr.Zero)
            {
                Marshal.Release(interopPointer);
            }
            if (className != IntPtr.Zero)
            {
                WindowsDeleteString(className);
            }
        }
    }

    public static IDirect3DDevice CreateWinRtDevice(ID3D11Device device)
    {
        using var dxgiDevice = device.QueryInterface<IDXGIDevice>();
        var hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.NativePointer, out var inspectable);
        hr.ThrowOnFailure();
        try
        {
            return MarshalInterface<IDirect3DDevice>.FromAbi(inspectable);
        }
        finally
        {
            Marshal.Release(inspectable);
        }
    }

    public static ID3D11Texture2D GetTexture(IDirect3DSurface surface)
    {
        var access = surface.As<IDirect3DDxgiInterfaceAccess>();
        var textureGuid = D3D11Texture2DGuid;
        var pointer = access.GetInterface(in textureGuid);
        if (pointer == IntPtr.Zero)
        {
            throw new InvalidOperationException("IDirect3DDxgiInterfaceAccess returned a null texture pointer.");
        }
        return new ID3D11Texture2D(pointer);
    }

    [DllImport("combase.dll", CharSet = CharSet.Unicode)]
    private static extern int WindowsCreateString(
        string sourceString,
        uint length,
        out IntPtr hstring);

    [DllImport("combase.dll")]
    private static extern int WindowsDeleteString(IntPtr hstring);

    [DllImport("combase.dll")]
    private static extern int RoGetActivationFactory(
        IntPtr activatableClassId,
        ref Guid iid,
        out IntPtr factory);

    [DllImport("d3d11.dll")]
    private static extern int CreateDirect3D11DeviceFromDXGIDevice(
        IntPtr dxgiDevice,
        out IntPtr graphicsDevice);

    [ComImport]
    [Guid("3628E81B-3CAC-4C60-B7F4-23CE0E0C3356")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IGraphicsCaptureItemInterop
    {
        [PreserveSig]
        int CreateForWindow(IntPtr window, ref Guid iid, out IntPtr result);

        [PreserveSig]
        int CreateForMonitor(IntPtr monitor, ref Guid iid, out IntPtr result);
    }

    [ComImport]
    [Guid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    [ComVisible(true)]
    private interface IDirect3DDxgiInterfaceAccess
    {
        IntPtr GetInterface(in Guid iid);
    }

    private static void ThrowOnFailure(this int hr)
    {
        if (hr < 0)
        {
            Marshal.ThrowExceptionForHR(hr);
        }
    }
}
