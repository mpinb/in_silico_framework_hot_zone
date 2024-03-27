var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    // Set the content style overflow of all other contents to hidden
    var allCollapsibles = document.getElementsByClassName("collapsible");
    for (var j = 0; j < allCollapsibles.length; j++) {
      if (allCollapsibles[j] !== this) {
        var content = allCollapsibles[j].nextElementSibling;
        content.style.height = "0";
        content.style.display = "hidden";
      }
    }

    // extend or collapse current collapsible
    var extended_height = "300%";
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

var divA = document.querySelector("#row2");
var arrow  = document.querySelector("#arrow");

function extendConnector() {
  var posnA = {
    x: divA.offsetLeft + divA.offsetWidth / 3 - divA.offsetWidth/6,
    y: divA.offsetTop 
  };

  var posnB = {
    y: divA.offsetTop + coll[0].offsetTop
  };

  var dStr =
      "M" +
      (posnA.x) + "," + (posnA.y) + " " +
      "V" +
      (posnB.y)
  arrow.setAttribute("d", dStr);
};

function collapseConnector() {
  var posnA = {
    x: divA.offsetLeft + divA.offsetWidth / 3 - divA.offsetWidth/6,
    y: divA.offsetTop
  };

  var posnB = {
    y: divA.children[0].offsetTop
  };

  var dStr =
      "M" +
      (posnA.x      ) + "," + (posnA.y) + " " +
      "V" +
      (posnB.y)
  arrow.setAttribute("d", dStr);
}


collapseConnector();
