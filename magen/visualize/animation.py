import matplotlib.pyplot as plt
def animation_simple(self, max_frame=False):
    fig = plt.figure(tight_layout=True, figsize=[12, 8])
    gs = gridspec(3, 2)
    ax_up, ax_up_right, ax_down = [fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[0:2, 1]),
                                   fig.add_subplot(gs[2, :])]

    if max_frame is False:
        n_iter_total = len(self.probed_points)
    else:
        n_iter_total = max_frame

    axins = self.visualizer.plot_background_zoom(ax=ax_up, fig=fig)
    up_right_colorbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=None), ax=ax_up_right)
    # fig.colorbar(contf, ax=ax_up_right)
    ax_up.set_ylabel(r'$J_z$', fontsize=16)
    ax_up.set_xlabel(r'$J_x = J_y$', fontsize=16)

    ax_up.set_title(r'Landscape of $GoF$ ''\n''Predicted by GP after {} iteration'.format(len(self.probed_value)), fontsize=16)

    probed_scatter = ax_up.scatter([], [], label='History Queries', color='white', alpha=0.4)
    probed_scatter_in = axins.scatter([], [], color='white', alpha=0.4)
    newest_prob_point = ax_up.scatter([], [], label='New Aquisition', color='white', alpha=1)
    cross_v, = ax_up.plot([], [], color='white', alpha=0.5, lw=1, ls='-')
    cross_h, = ax_up.plot([], [], color='white', alpha=0.5, lw=1, ls='-')

    value_plot = ax_down.plot(np.arange(1, n_iter_total + 1), self.probed_value[0:n_iter_total],
                              'o', fillstyle='none', label='Aquired Value per Iteration')[0]
    lowest_value_plot = ax_down.plot([], [], label='Max Value')[0]
    ax_up_xlim = ax_up.get_xlim()
    ax_up_ylim = ax_up.get_ylim()

    def init():
        ax_down.set_xlabel('Iteration', fontsize=20)
        ax_down.set_xlim(0, n_iter_total + 1)
        ax_down.tick_params(labelsize=14)
        ax_up.legend(fontsize=15, loc='lower right')
        ax_down.legend(fontsize=15, loc='lower right')
        #     value_plot.set_data([], [])
        return (probed_scatter, newest_prob_point, value_plot, lowest_value_plot)

    def update(frame):
        # print(frame)
        probed_scatter.set_offsets(self.probed_points[0:frame])
        probed_scatter_in.set_offsets(self.probed_points[0:frame])
        newest_prob_point.set_offsets(self.probed_points[frame])

        ax_up_right.clear()

        meancontf = self.plot_gp(ax=ax_up_right, predictor=self.record[frame]._gp, mode='mean')
        up_right_colorbar.on_mappable_changed(meancontf)


        cross_h.set_data(np.linspace(ax_up_xlim[0], ax_up_xlim[1], 100), np.ones(100) * self.probed_points[frame][1])
        cross_v.set_data(np.ones(100) * self.probed_points[frame][0], np.linspace(ax_up_ylim[0], ax_up_ylim[1], 100))

        value_plot.set_data(np.arange(1, frame + 2), self.probed_value[0:frame + 1])
        # print(type(lowest_value_plot), type(self.probed_max_value))
        lowest_value_plot.set_data(np.arange(1, frame + 2), self.probed_max_value[0:frame + 1])
        return (probed_scatter, newest_prob_point, value_plot, lowest_value_plot, cross_h, cross_v)

    ani = FuncAnimation(fig, update, frames=range(n_iter_total),
                        init_func=init, blit=True)
    writer = FFMpegWriter(fps=8, metadata=dict(artist='Me'), bitrate=1800)
    # HTML(ani.to_html5_video())
    HTML(ani.to_jshtml())
    # ani.save("Result_BGP.mp4", writer=writer)
    return ani, writer