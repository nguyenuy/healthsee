$( document ).ready(function() {
	$('#submit_button').click(function() {
	   $.ajax(window.location.href +"/report_by_zipcode/" + inputZipCode.text).done(function (reply) {
		  $('#dashboard').html(reply);
	   });
	});
});