import holoviews as hv
import pandas as pd
from bokeh.plotting import output_file, show

hv.extension('bokeh')

def graph_peaks_bokeh(pa):
    df_signal = pd.DataFrame({
        'time': pa.time_numeric,
        'voltage': pa.voltage
    })

    curve = hv.Curve(df_signal, 'time', 'voltage').opts(
        width=900, height=400, tools=['hover', 'box_zoom', 'wheel_zoom', 'pan']
    )

    peaks_df = pd.DataFrame({
        'time': pa.time_numeric[pa.df_peaks['peak_index']],
        'voltage': pa.voltage[pa.df_peaks['peak_index']]
    })

    peaks_scatter = hv.Points(peaks_df, ['time', 'voltage']).opts(color='red', size=5)

    plot = curve * peaks_scatter

    output_file("Plots/plot.html")
    show(hv.render(plot, backend='bokeh'))
    return plot
