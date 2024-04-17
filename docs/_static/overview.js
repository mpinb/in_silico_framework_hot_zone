var buttons = document.getElementsByClassName("organigram-collapsible");
var contents = document.getElementsByClassName("organigram-collapsible-content")
var i;

for (i = 0; i < buttons.length; i++) {
  buttons[i].addEventListener("click", function() {
    this.classList.toggle("active");
    for (var j = 0; j < buttons.length; j++) {    
        if (buttons[j] !== this) {
            // Set the content style overflow of all other contents to hidden
            collapseContent(j);
          } else {
            // Expand or collapse the content of whatever button was clicked 
            var extended_height = "300px";  // should be an absolute number to work in all browsers
            if (contents[j].style.height === extended_height) {
              collapseContent(j)
            } else {
              expandContent(j, extended_height)
            };
          }
        }

    
  })
}

function collapseContent(j) {
  // collapse content that corresponds to index j
  contents[j].style.height = "0";
  contents[j].style.display = "block";
}

function expandContent(j, height) {
  // expand content that corresponds to index j
  contents[j].style.height = height;
  contents[j].style.display = "block";
}
