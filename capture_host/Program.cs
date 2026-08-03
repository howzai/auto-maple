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

    private static int Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.WriteLine("MapleCaptureHost - Windows Graphics Capture backend");

        if (!OperatingSystem.IsWindowsVersionAtLeast(10, 0, 18362))
        {
            Console.Error.WriteLine("Windows 10 1903 (build 18362) or newer is required.");
            return 2;
        }

        var window = FindTargetWindow();
        if (window.Handle == IntPtr.Zero)
        {
            Console.Error.WriteLine("Target game window was not found.");
            return 3;
        }

        Console.WriteLine($"Target window found: '{window.Title}' HWND=0x{window.Handle.ToInt64():X}");
        Console.WriteLine("Capture pipeline scaffold is ready. The next implementation step is GraphicsCaptureItem + D3D11 frame delivery.");
        return 0;
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
