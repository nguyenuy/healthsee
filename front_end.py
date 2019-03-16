#import statements
import os
import numpy as np
#bokeh - plotting capabilities
from bokeh.plotting import figure
from bokeh.embed import components
#Flask - web serving capabilities
from flask import Flask, render_template
#zipfile - data storage utilities
import zipfile

def unpack_data(path_to_zip_file, directory_to_extract_to):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

# Create the main plot
def create_figure():
    N = 40
    x = np.random.random(size=N) * 100
    y = np.random.random(size=N) * 100
    radii = np.random.random(size=N) * 1.5
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)
    ]
    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    p = figure(tools=TOOLS)
    p.scatter(x, y, radius=radii,
          fill_color=colors, fill_alpha=0.6,
          line_color=None)
    return p

#Make the app
app = Flask(__name__)
    
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/health_report")
def health_report():
    #create two vectors
    plot = create_figure()
    script, div = components(plot)
    return render_template("health_report.html", plot = [div, script])
                           
if __name__ == "__main__":
    #unpack the data
    if not os.path.exists("data"):
        os.mkdir("data")
        print("directory created!")
    #extract the data
    print("unpacking_data...")
    unpack_data("data.zip", "data")
    #run the app
    print("running the app.")
    app.run()
