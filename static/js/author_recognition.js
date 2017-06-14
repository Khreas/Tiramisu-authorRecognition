/* Radar chart design created by Nadieh Bremer - VisualCinnamon.com */

////////////////////////////////////////////////////////////// 
//////////////////////// Set-Up ////////////////////////////// 
////////////////////////////////////////////////////////////// 

var first_time = true;

var margin = {top: 150, right: 150, bottom: 50, left: 150},
	width = Math.min(600, window.innerWidth - 10) - margin.left - margin.right,
	height = Math.min(width, window.innerHeight - margin.top - margin.bottom - 20);
		
////////////////////////////////////////////////////////////// 
////////////////////////// Data ////////////////////////////// 
////////////////////////////////////////////////////////////// 

var data_full;

$.ajax({
		  url: "../static/resources/json/probabilities.json",
		  dataType: 'json',
		  async: false,
		  success: function(data) {
		    data_full = data;
		  }
		});

////////////////////////////////////////////////////////////// 
//////////////////// Draw the Chart ////////////////////////// 
////////////////////////////////////////////////////////////// 

var color = d3.scale.ordinal()
	.range(["#EDC951","#CC333F","#00A0B0"]);
	
var radarChartOptions = {
  w: width,
  h: height,
  margin: margin,
  maxValue: 0.5,
  levels: 5,
  roundStrokes: true,
  color: color
};

for(var i = 0 ; i < data_full[0][0].length ; i++)
{
	data_full[0][0][i]['value'] = 0.0;
}

//Call function to draw the Radar chart
RadarChart(".radarChart", data_full[0], radarChartOptions);

var authors_info;

$.ajax({
  url: "../static/resources/json/authors.json",
  dataType: 'json',
  async: false,
  success: function(data) {
    authors_info = data;
  }
});

document.getElementById("image_holder").style.visibility='hidden';
document.getElementById("text_author").style.visibility='hidden';
document.getElementById("date_name_auth").style.visibility='hidden';
document.getElementById("percentage_author").style.visibility='hidden';

function readTextFile(file)
{
    var rawFile = new XMLHttpRequest();
    var allText;
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                allText = rawFile.responseText;
            }
        }
    }
    rawFile.send(null);
    return allText;
}

function onClickInit() {

	$('.btn').button('loading');

	$.ajax({
	        type:'POST',
	        data:
	        {
	            analysis: "launched"
	        }
	    });

	if(data_full[1]['state'] == '' || data_full[1]['state'] == 'stop') {

		var text_read = readTextFile("../static/resources/text_processed.txt");

		document.getElementById("image_holder").style.visibility='hidden';
		document.getElementById("text_author").style.visibility='hidden';
		document.getElementById("date_name_auth").style.visibility='hidden';
		document.getElementById("percentage_author").style.visibility='hidden';

		wrapper(text_read, true);
	}

}

function wrapper(text_read, restart=false)
{

    $.ajax({
		  url: "../static/resources/json/probabilities.json",
		  dataType: 'json',
		  success: function(data) {
		    data_full = data;
		  }
		});

    if(data_full[1]["state"] != "stop" || restart == true) {

	scroller(text_read);
	RadarChart(".radarChart", data_full[0], radarChartOptions);

	wrapperRefresh = setTimeout(function(){wrapper(text_read)}, 100);

	} else {
		
		var index_auth = data_full[1]['predicted author'];

		var max_percentage = 0.0;

		for(var i = 0; i < data_full[0][0].length ; i++){
			console.log(data_full[0][0][i]['value']);
			if(data_full[0][0][i]['value'] > max_percentage){
				max_percentage = data_full[0][0][i]['value'];
			}
		}

		document.getElementById("image_auth").src='../static/resources/'+authors_info[index_auth]['author']+'.jpg';
		document.getElementById("date_name_auth").innerHTML = '<b>' + authors_info[index_auth]['author'] + '</b><br>' + authors_info[index_auth]['date'];
		document.getElementById("text_author").innerHTML = authors_info[index_auth]['info'];

		document.getElementById("percentage_author").innerHTML = (max_percentage * 100).toFixed(2) + '%';

		document.getElementById("image_holder").style.visibility='visible';
		document.getElementById("text_author").style.visibility='visible';
		document.getElementById("date_name_auth").style.visibility='visible';
		document.getElementById("percentage_author").style.visibility='visible';
		RadarChart(".radarChart", data_full[0], radarChartOptions);

		$('.btn').button('reset');

	}
}


function scroller(text){

	var index = data_full[1]['current_index'];
	var nb_sample = data_full[1]['max_len'];
	var real_len = text.length;

	var limit = (index / nb_sample) * real_len;

	text = text.substring(0, limit) + "<font color='blue'>" + text.substring(limit, limit + 376), + "</font>" + text.substring( limit + 376, text.length) 

	document.getElementById("text_processed").innerHTML = text;
	document.getElementById('text_processed').scrollTop = document.getElementById('text_processed').scrollHeight;
}