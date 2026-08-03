using System.Runtime.InteropServices;
using System.Text;

namespace MapleCaptureHost;

internal static class Program
{
    private static readonly string[] TitleKeywords =
    {
        "新楓之谷：經典版",
        "新楓之谷",
        "MapleStory"
    };

    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.WriteLine("MapleCaptureHost - Windows Graphics Capture backend");

        if (!OperatingSystem.IsWindowsVersionAtLeast(10, 0, 18362))
        {
            Console.Error.WriteLine("Windows 10 1903 (build 18362) or newer is required.");
            return 2;
        }

        using var shutdown = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            shutdown.Cancel();
        };

        using var output = new SharedFrameBuffer();
        if (args.Any(arg => string.Equals(arg, "--test-pattern", StringComparison.OrdinalIgnoreCase)))
        {
            Console.WriteLine("Publishing deterministic 640x360 BGRA test frames at 30 FPS.");
            Console.WriteLine("Press Ctrl+C to stop.");
            try
            {
                await TestPatternSource.RunAsync(output, shutdown.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Normal Ctrl+C shutdown.
            }
            return 0;
        }

        var window = FindTargetWindow();
        if (window.Handle == IntPtr.Zero)
        {
            Console.Error.WriteLine("Target game window was not found.");
            return 3;
        }

        Console.WriteLine($"Target window found: '{window.Title}' HWND=0x{window.Handle.ToInt64():X}");
        Console.WriteLine("Shared-frame transport is ready.");
        Console.WriteLine("GraphicsCaptureItem + D3D11 frame-copy implementation is the remaining host step.");
        return 4;
    }

    private static WindowMatch FindTargetWindow()
    {
        var matches = new List<WindowMatch>();
        EnumWindows((handle, _) =>
        {
            if (!IsWindowVisible(handle))
            {
                return true;
            }

            var length = GetWindowTextLengthW(handle);
            if (length <= 0)
            {
                return true;
            }

            var builder = new StringBuilder(length + 1);
            _ = GetWindowTextW(handle, builder, builder.Capacity);
            var title = builder.ToString().Trim();
            if (TitleKeywords.Any(keyword => title.Contains(keyword, StringComparison.OrdinalIgnoreCase)))
            {
                matches.Add(new WindowMatch(handle, title, GetWindowArea(handle)));
            }

            return true;
        }, IntPtr.Zero);

        return matches
            .OrderByDescending(match => match.Area)
            .FirstOrDefault();
    }

    private static long GetWindowArea(IntPtr handle)
    {
        return GetWindowRect(handle, out var rect)
            ? Math.Max(0, rect.Right - rect.Left) * (long)Math.Max(0, rect.Bottom - rect.Top)
            : 0;
    }

    private readonly record struct WindowMatch(IntPtr Handle, string Title, long Area);

    private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowTextLengthW(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowTextW(IntPtr hWnd, StringBuilder text, int maxCount);

    [DllImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetWindowRect(IntPtr hWnd, out Rect rect);

    [StructLayout(LayoutKind.Sequential)]
    private struct Rect
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
}
