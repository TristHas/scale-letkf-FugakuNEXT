import holoviews as hv
import hvplot.pandas  # noqa: F401  ensures hvplot extension registered

def plot_res_chunk(res_chunk):
    return hv.Layout([
        hv.Overlay([res_chunk[(res_chunk["dtype"]==dtype) &\
                    (res_chunk["backend"]==backend)].hvplot(x="chunk_size", y="to_letkf")
                    for dtype in res_chunk["dtype"].unique()])
        for backend in res_chunk["backend"].unique()
    ]).cols(1)

def plot_res(res):
    all_plots = []
    
    for backend in ["torch", "triton"]:
        plots = []
        for dt in res["dtype"].unique():
            r = res[res.dtype==dt]
            p = r[r["backend"]==backend].sort_values(by="N").hvplot(x="N", y="to_scale", label=dt)
            df = r[r["backend"]==backend].sort_values(by="N")
        
            line = df.hvplot(
                x="N", y="to_scale",
                label=dt,
                line_width=2
            )
            points = df.hvplot.scatter(
                x="N", y="to_scale",
                size=60,
                alpha=0.8
            )
            
            p = line * points
            plots.append(p)
    
        plots = hv.Overlay(plots).opts(xlim=(0, res[["to_scale", "N"]].dropna()["N"].max()),
                                       ylim=(0, res.dropna()["to_scale"].max()),
                                       title=backend)
        all_plots.append(plots)
    return hv.Layout(all_plots).cols(1)
