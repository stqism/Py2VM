var options={bg:'#03A9F4',id:'mynano'};var nanobar=new Nanobar(options);nanobar.go(100);jQuery(window).load(function(){nanobar.go(0)});jQuery(window).unload(function(){nanobar.go(100)});jQuery(window).click(function(){nanobar.go(100)});
window.onhashchange = newpage;
if (location.hash != '') {
  newpage();
}

$('#body').on("swiperight",function(){
  history.back()
});

$('#body').on("swipeleft",function(){
  history.forward()	
});

function newpage() {
    if (location.hash == '') {
      $('#tag').load("index.html #tag");
      $('#list').load("index.html #list");
    } else {

      $('#tag').load(location.hash.slice( 1 ) + ".html #tag");
      $('#list').load(location.hash.slice( 1 ) + ".html #list");
    }
};
