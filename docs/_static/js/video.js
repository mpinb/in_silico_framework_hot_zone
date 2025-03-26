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


    // Play video after the first user interaction
    function enableAutoplayOnInteraction() {
        videoPlayer.play().catch((error) => {
            console.warn("Autoplay failed after user interaction:", error);
        });

        // Remove event listeners after the first interaction
        document.removeEventListener("click", enableAutoplayOnInteraction);
        document.removeEventListener("keydown", enableAutoplayOnInteraction);
        document.removeEventListener("mousemove", enableAutoplayOnInteraction);
    }

    // Add event listeners for user interaction
    document.addEventListener("click", enableAutoplayOnInteraction);
    document.addEventListener("keydown", enableAutoplayOnInteraction);
    document.addEventListener("mousemove", enableAutoplayOnInteraction);

    // Initial check
    updateVideoSource();

    // Listen for theme changes
    const observer = new MutationObserver(updateVideoSource);
    observer.observe(htmlElement.lastChild, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
});