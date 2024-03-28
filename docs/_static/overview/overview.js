var coll = document.getElementsByClassName("organigram-collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    // Set the content style overflow of all other contents to hidden
    var allCollapsibles = document.getElementsByClassName("organigram-collapsible");
    for (var j = 0; j < allCollapsibles.length; j++) {
      if (allCollapsibles[j] !== this) {
        var content = allCollapsibles[j].nextElementSibling;
        content.style.height = "0";
        content.style.display = "hidden";
      }
    }

    // extend or collapse current collapsible
    var extended_height = "300px";  // should be an absolute number to work in all browsers
    var content = this.nextElementSibling;
    if (content.style.height === extended_height) {
      content.style.height = "0";
      content.style.overflow = "hidden";
      extendConnector();
      
    } else {
      content.style.height = extended_height;
      content.style.display = "block";
      collapseConnector();
    };
    
  })
}

