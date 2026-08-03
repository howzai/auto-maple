using System.Diagnostics;

namespace MapleCaptureHost;

internal static class TestPatternSource
{
    public static async Task RunAsync(SharedFrameBuffer output, CancellationToken cancellationToken)
    {
        const int width = 640;
        const int height = 360;
        const int stride = width * SharedFrameBuffer.BytesPerPixel;
        var pixels = new byte[stride * height];
        var stopwatch = Stopwatch.StartNew();
        var frame = 0;

        while (!cancellationToken.IsCancellationRequested)
        {
            Fill(pixels, width, height, stride, frame++);
            output.Publish(pixels, width, height, stride, DateTime.UtcNow.Ticks);

            var target = TimeSpan.FromMilliseconds(frame * (1000.0 / 30.0));
            var remaining = target - stopwatch.Elapsed;
            if (remaining > TimeSpan.Zero)
            {
                await Task.Delay(remaining, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static void Fill(Span<byte> pixels, int width, int height, int stride, int frame)
    {
        pixels.Clear();
        for (var y = 0; y < height; y++)
        {
            var row = pixels.Slice(y * stride, stride);
            for (var x = 0; x < width; x++)
            {
                var offset = x * 4;
                row[offset + 0] = (byte)((x + frame * 2) & 0xFF); // B
                row[offset + 1] = (byte)((y * 2 + frame) & 0xFF); // G
                row[offset + 2] = (byte)((x + y + frame * 3) & 0xFF); // R
                row[offset + 3] = 255;
            }
        }

        var boxX = 30 + frame % Math.Max(1, width - 90);
        var boxY = 70;
        for (var y = boxY; y < Math.Min(height, boxY + 44); y++)
        {
            var row = pixels.Slice(y * stride, stride);
            for (var x = boxX; x < Math.Min(width, boxX + 60); x++)
            {
                var offset = x * 4;
                row[offset + 0] = 0;
                row[offset + 1] = 255;
                row[offset + 2] = 255;
                row[offset + 3] = 255;
            }
        }
    }
}
