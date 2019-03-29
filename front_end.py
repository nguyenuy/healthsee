#import statements
import os
import numpy as np
#bokeh - plotting capabilities
from bokeh.plotting import figure, show, output_file
from bokeh.embed import components
#Flask - web serving capabilities
from flask import Flask, render_template
#health_data imports
import health_data
from bokeh.tile_providers import CARTODBPOSITRON



# Create the main plot
def create_normalized_score_figure(zipcodes):
#    scores = []
#    for i in range(0, len(zipcodes)):
#        score = health_data.get_total_normalized_health_score(zipcodes[i])
#        scores.append(score)
#    colors = [
#        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)
#    ]
#    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
#    p = figure(x_range=zipcodes, tools=TOOLS)
#    p.bar(zipcodes, scores, 
#          fill_color=colors, fill_alpha=0.6,
#          line_color=None)
    p = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
           x_axis_type="mercator", y_axis_type="mercator")
    p.add_tile(CARTODBPOSITRON)
    return p

#create a new plot based n zip code
def generate_report_by_zip_code(zip_code):
    return create_figure()

#Make the app
app = Flask(__name__)
    
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/report_by_zipcode/<zipcode>")
def report_by_zipcode(zipcode):
    #get health report for this zip code
    print(zipcode)
    my_report = health_data.calculate_health_score(zipcode)
    print(my_report)
    #get my normalized health score
    my_score = health_data.get_total_normalized_health_score(zipcode)
    #get every zip code within a 25 mile radius
    zipcodes = health_data.get_closest_zipcodes(zipcode)
    #get their health reports
    neighbors_health_reports = []
    for zipcode in zipcodes:
        report = health_data.calculate_health_score(zipcode)
        neighbors_health_reports.append(report)
    #get a plot of the scores
    plot = create_normalized_score_figure(zipcodes)
    script, div = components(plot)
    return render_template("dashboard.html", plot = [div, script], my_report = my_report, neighbor_report = neighbors_health_reports)
    
@app.route("/health_report")
def health_report():
    return report_by_zipcode(30033)

if __name__ == "__main__":
    #run the app
    print("running the app.")
    #health_data.calculate_health_score(30084)
    app.run()
