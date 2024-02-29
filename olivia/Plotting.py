import tables as tb
import numpy as np
import matplotlib.pyplot as plt



# h5dump -n file.h5 to check the group and dataset

# Groupnames are: [/DST/Events, /Filters/s12_selector, 
# /MC/configuration, /MC/event_mapping, /MC/hits, /MC/particles, /MC/sns_positions,
# /MC/sns_response,/Run/eventMap /Run/events, /Run/runInfo
       
def dorothea_histogram(filename, groupname):
    with tb.open_file(filename, mode='r') as h5file:
        group = h5file.get_node(groupname)
        
        num_columns = len(group.dtype.names)
        num_rows = (num_columns - 1) // 3 + 1
        num_cols = min(num_columns, 3)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
        
        for i, column in enumerate(group.dtype.names):
            row = i // num_cols
            col = i % num_cols
 
            data = group.col(column)
            
            ax = axes[row, col] if num_rows > 1 else axes[col]
            
            max_val = np.max(data)
            min_val = np.min(data)
            if max_val > 10 * np.mean(data):
                min_val = max(1e-10, min_val)
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
                ax.set_yscale('log')
            else:
                bins = np.linspace(min_val, max_val, 50)
                
            ax.hist(data, bins=bins, histtype='step', linewidth=1.5, color='black')
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel("Entries")
            
            total_entries = len(data)
            mean = np.mean(data)
            rms = np.sqrt(np.mean(np.square(data - mean)))
            ax.annotate(f'Entries= {total_entries}\nMean= {mean:.2f}\nRMS= {rms:.2f}',
                        xy=(1, 1), xycoords='axes fraction',
                        xytext=(-5, -5), textcoords='offset points',
                        fontsize=10, ha='right', va='top')
            ax.grid(alpha=.5)
        
        for i in range(num_columns, num_rows * num_cols):
            axes.flatten()[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        
        
        
# Following works very nice, this is so you dont have to write the column you want a histo of
# Groupnames are: [/DST/Events, /Filters/s12_selector, /RECO/Events, /Run/events, /Run/runInfo]   

def sophronia_histogram(filename, groupname):
    with tb.open_file(filename, mode='r') as h5file:
        group = h5file.get_node(groupname)
        
        num_columns = len(group.dtype.names)
        num_rows = (num_columns - 1) // 3 + 1
        num_cols = min(num_columns, 3)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
        
        for i, column in enumerate(group.dtype.names):
            row = i // num_cols
            col = i % num_cols
 
            data = group.col(column)
            
            ax = axes[row, col] if num_rows > 1 else axes[col]

            max_val = np.max(data)
            min_val = np.min(data)
            if max_val > 10 * np.mean(data):
                min_val = max(1e-10, np.nanmin(data[np.isfinite(data)]))
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
                ax.set_yscale('log')
            else:
                bins = np.linspace(min_val, max_val, 50)
                
            ax.hist(data, bins=bins, histtype='step', linewidth=1.5, color='black')
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel("Entries")
            
            total_entries = len(data)
            mean = np.mean(data)
            rms = np.sqrt(np.mean(np.square(data - mean)))
            ax.annotate(f'Entries= {total_entries}\nMean= {mean:.2f}\nRMS= {rms:.2f}',
                        xy=(1, 1), xycoords='axes fraction',
                        xytext=(-5, -5), textcoords='offset points',
                        fontsize=10, ha='right', va='top')
            ax.grid(alpha=.5)
        
        for i in range(num_columns, num_rows * num_cols):
            axes.flatten()[i].axis('off')
        
        plt.tight_layout()
        plt.show()


# Groupnames are: [/CHITS/highTh, /DST/Events, /Filters/high_th_select, /Filters/topology_select /MC/hits, /Summary/Events, /Tracking/Tracks]

def esmeralda_histogram(filename, groupname):
    with tb.open_file(filename, mode='r') as h5file:
        group = h5file.get_node(groupname)
        
        num_columns = len(group.dtype.names)
        num_rows = (num_columns - 1) // 3 + 1
        num_cols = min(num_columns, 3)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
        
        for i, column in enumerate(group.dtype.names):
            row = i // num_cols
            col = i % num_cols
 
            data = group.col(column)
            
            ax = axes[row, col] if num_rows > 1 else axes[col]

            max_val = np.max(data)
            min_val = np.min(data)
            if max_val > 10 * np.mean(data):
                min_val = max(1e-10, min_val)
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
                ax.set_yscale('log')
            else:
                bins = np.linspace(min_val, max_val, 50)
                
            ax.hist(data, bins=bins, histtype='step', linewidth=1.5, color='black')
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel("Entries")
            
            total_entries = len(data)
            mean = np.mean(data)
            rms = np.sqrt(np.mean(np.square(data - mean)))
            ax.annotate(f'Entries= {total_entries}\nMean= {mean:.2f}\nRMS= {rms:.2f}',
                        xy=(1, 1), xycoords='axes fraction',
                        xytext=(-5, -5), textcoords='offset points',
                        fontsize=10, ha='right', va='top')
            ax.grid(alpha=.5)
        
        for i in range(num_columns, num_rows * num_cols):
            axes.flatten()[i].axis('off')
        
        plt.tight_layout()
        plt.show()
            

###############


# import numpy as np
# import matplotlib.pyplot as plt
# import tables as tb

# from invisible_cities.reco import tbl_functions as tbl
# from invisible_cities.core.core_functions import shift_to_bin_centers, weighted_mean_and_std
# from olivia.histos import Histogram, HistoManager

# def hist_writer_var(file, *, compression='ZLIB4'):
#     def write_hist(group_name, table_name, entries, bins, out_of_range, errors, labels, scales):
#         try:
#             hist_group = getattr(file.root, group_name)
#         except tb.NoSuchNodeError:
#             hist_group = file.create_group(file.root, group_name)

#         if table_name in hist_group:
#             raise ValueError(f"Histogram {table_name} already exists")

#         vlarray = file.create_vlarray(hist_group, f'{table_name}_bins', atom=tb.Float64Atom(shape=()), filters=tbl.filters(compression))
#         for ibin in bins:
#             vlarray.append(ibin)
#         _add_carray(hist_group, table_name, entries)
#         _add_carray(hist_group, f'{table_name}_outRange', out_of_range)
#         _add_carray(hist_group, f'{table_name}_errors', errors)
#         file.create_array(hist_group, f'{table_name}_labels', labels)
#         file.create_array(hist_group, f'{table_name}_scales', scales)

#     def _add_carray(hist_group, table_name, var):
#         array_atom = tb.Atom.from_dtype(var.dtype)
#         array_shape = var.shape
#         entry = file.create_carray(hist_group, table_name, atom=array_atom, shape=array_shape, filters=tbl.filters(compression))
#         entry[:] = var

#     return write_hist

# def get_histograms_from_file(file_input, group_name='HIST'):
#     histo_manager = HistoManager()

#     def name_selection(x):
#         selection = (('bins' not in x)
#                      and ('labels' not in x)
#                      and ('errors' not in x)
#                      and ('outRange' not in x)
#                      and ('scales' not in x))
#         return selection

#     with tb.open_file(file_input, "r") as h5in:
#         histogram_list = []
#         group = getattr(h5in.root, group_name)
#         for histoname in filter(name_selection, group._v_children):
#             entries = np.array(getattr(group, histoname))
#             bins = getattr(group, f'{histoname}_bins')[:]
#             out_range = getattr(group, f'{histoname}_outRange')[:]
#             errors = np.array(getattr(group, f'{histoname}_errors'))
#             labels = getattr(group, f'{histoname}_labels')[:]
#             labels = [str(lab)[2:-1].replace('\\\\', '\\') for lab in labels]
#             try:
#                 scale = getattr(group, f'{histoname}_scales')[:]
#                 scale = [str(scl)[2:-1].replace('\\\\', '\\') for scl in scale]
#             except tb.NoSuchNodeError:
#                 scale = ["linear"]

#             histogram = Histogram(histoname, bins, labels, scale)
#             histogram.data = entries
#             histogram.out_range = out_range
#             histogram.errors = errors
#             histogram.scale = scale

#             histogram_list.append(histogram)

#     return HistoManager(histogram_list)

# def plot_histograms_from_file(histofile, histonames='all', group_name='HIST', plot_errors=False, out_path=None, reference_histo=None):
#     histograms = get_histograms_from_file(histofile, group_name)
#     if reference_histo:
#         reference_histo = get_histograms_from_file(reference_histo, group_name)

#     if histonames == 'all':
#         histonames = histograms.histos

#     if out_path is None:
#         n_histos = len(histonames)
#         n_columns = min(3, n_histos)
#         n_rows = int(np.ceil(n_histos / n_columns))

#         fig, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 6 * n_rows))

#     for i, histoname in enumerate(histonames):
#         if out_path:
#             fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#         else:
#             ax = axes.flatten()[i] if isinstance(axes, np.ndarray) else axes

#         if reference_histo:
#             if len(reference_histo[histoname].bins) == 1:
#                 plot_histogram(reference_histo[histoname], ax=ax, plot_errors=plot_errors, normed=True, draw_color='red', stats=False)

#         plot_histogram(histograms[histoname], ax=ax, plot_errors=plot_errors, normed=True)

#         if out_path:
#             fig.tight_layout()
#             fig.savefig(out_path + histoname + '.png')
#             fig.clf()
#             plt.close(fig)
#     if out_path is None:
#         fig.tight_layout()

# def plot_histogram(histogram, ax=None, plot_errors=False, draw_color='black', stats=True, normed=True):
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))

#     bins = histogram.bins
#     out_range = histogram.out_range
#     labels = histogram.labels
#     title = histogram.title
#     scale = histogram.scale
#     if plot_errors:
#         entries = histogram.errors
#     else:
#         entries = histogram.data

#     if len(bins) == 1:
#         ax.hist(shift_to_bin_centers(bins[0]), bins[0], weights=entries, histtype='step', edgecolor=draw_color, linewidth=1.5, density=normed)
#         ax.grid(True)
#         ax.set_axisbelow(True)
#         ax.set_ylabel("Entries", weight='bold', fontsize=20)
#         ax.set_yscale(scale[0])

#         if stats:
#             entries_string = f'Entries = {np.sum(entries):.0f}\n'
#             out_range_string = 'Out range (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0, 0], np.sum(entries)),
#                                                                            get_percentage(out_range[1, 0], np.sum(entries)))

#             if np.sum(entries) > 0:
#                 mean, std = weighted_mean_and_std(shift_to_bin_centers(bins[0]), entries, frequentist=True, unbiased=True)
#             else:
#                 mean, std = 0, 0

#             ax.annotate(entries_string +
#                         'Mean = {0:.2f}\n'.format(mean) +
#                         'RMS = {0:.2f}\n'.format(std) +
#                         out_range_string,
#                         xy=(0.99, 0.99),
#                         xycoords='axes fraction',
#                         fontsize=11,
#                         weight='bold',
#                         color='black',
#                         horizontalalignment='right',
#                         verticalalignment='top')

#     elif len(bins) == 2:
#         ax.pcolormesh(bins[0], bins[1], entries.T)
#         ax.set_ylabel(labels[1], weight='bold', fontsize=20)

#         if stats:
#             entries_string = f'Entries = {np.sum(entries):.0f}\n'
#             out_range_stringX = 'Out range X (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0, 0], np.sum(entries)),
#                                                                               get_percentage(out_range[1, 0], np.sum(entries)))
#             out_range_stringY = 'Out range Y (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0, 1], np.sum(entries)),
#                                                                               get_percentage(out_range[1, 1], np.sum(entries)))

#             if np.sum(entries) > 0:
#                 meanX, stdX = weighted_mean_and_std(shift_to_bin_centers(bins[0]), np.sum(entries, axis=1), frequentist=True, unbiased=True)
#                 meanY, stdY = weighted_mean_and_std(shift_to_bin_centers(bins[1]), np.sum(entries, axis=0), frequentist=True, unbiased=True)
#             else:
#                 meanX, stdX = 0, 0
#                 meanY, stdY = 0, 0

#             ax.annotate(entries_string +
#                         'Mean X = {0:.2f}\n'.format(meanX) + 'Mean Y = {0:.2f}\n'.format(meanY) +
#                         'RMS X = {0:.2f}\n'.format(stdX) + 'RMS Y = {0:.2f}\n'.format(stdY) +
#                         out_range_stringX + '\n' + out_range_stringY,
#                         xy=(0.99, 0.99),
#                         xycoords='axes fraction',
#                         fontsize=11,
#                         weight='bold',
#                         color='white',
#                         horizontalalignment='right',
#                         verticalalignment='top')

#     elif len(bins) == 3:
#         ave = np.apply_along_axis(average_empty, 2, entries, shift_to_bin_centers(bins[2]))
#         ave = np.ma.masked_array(ave, ave < 0.00001)

#         img = ax.pcolormesh(bins[0], bins[1], ave.T)
#         cb = plt.colorbar(img, ax=ax)
#         cb.set_label(labels[2], weight='bold', fontsize=20)

#         for label in cb.ax.yaxis.get_ticklabels():
#             label.set_weight('bold')
#             label.set_fontsize(16)

#         ax.set_ylabel(labels[1], weight='bold', fontsize=20)
#     ax.set_xlabel(labels[0], weight='bold', fontsize=20)
#     ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))

#     for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#         label.set_fontweight('bold')
#         label.set_fontsize(16)
#     ax.xaxis.offsetText.set_fontsize(14)
#     ax.xaxis.offsetText.set_fontweight('bold')


# def average_empty(x, bins):
#     return np.average(bins, weights=x) if np.any(x > 0.) else 0.

# def get_percentage(a, b):
#     return 100 * a / b if b else -100
    


