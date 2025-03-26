document.addEventListener("DOMContentLoaded", function () {
    const videoSource = document.getElementById("video-source");
    const prefersDarkScheme = window.matchMedia("(prefers-color-scheme: dark)");

    function updateVideoSource() {
        if (prefersDarkScheme.matches) {
            videoSource.src = "_static/_videos/sequence_dark.mp4"; // Video for dark theme
        } else {
            videoSource.src = "_static/_videos/sequence.mp4"; // Video for light theme
        }
        videoSource.parentElement.load(); // Reload the video with the new source
    }

    // Initial check
    updateVideoSource();

    // Listen for theme changes
    prefersDarkScheme.addEventListener("change", updateVideoSource);
});