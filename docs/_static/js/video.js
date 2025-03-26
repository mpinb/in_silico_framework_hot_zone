document.addEventListener("DOMContentLoaded", function () {
    const videoPlayer = document.getElementById("video-player");
    const video = videoPlayer.getElementsByTagName("video")[0];
    const videoSource = document.getElementById("video-source");
    const htmlElement = document.documentElement; // The <html> element

    function updateVideoSource() {
        const theme = htmlElement.lastChild.getAttribute("data-md-color-scheme");
        if (theme === "slate") {
            videoSource.src = "_static/_videos/sequence_dark.mp4"; // Video for dark theme
        } else {
            videoSource.src = "_static/_videos/sequence.mp4"; // Video for light theme
        }
        videoSource.parentElement.load(); // Reload the video with the new source
    }

    // Initial check
    updateVideoSource();
    // Ensure the video is muted and autoplay works
    videoPlayer.muted = true; // Explicitly set muted
    // Ensure the video autoplays after the source is updated
    video.addEventListener("loadeddata", function () {
        video.play().catch((error) => {
            console.warn("Autoplay failed:", error);
        });
    });

    // Listen for theme changes
    const observer = new MutationObserver(updateVideoSource);
    observer.observe(htmlElement.lastChild, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
});