class NullArtist:
    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop

    def remove(self):
        return None


class NullCanvas:
    def draw_idle(self):
        return None

    def flush_events(self):
        return None


class NullFigure:
    def __init__(self):
        self.number = 0
        self.canvas = NullCanvas()

    def tight_layout(self):
        return None

    def align_ylabels(self):
        return None


class NullAxes:
    def __init__(self, figure=None):
        self.figure = figure if figure is not None else NullFigure()

    def add_patch(self, _patch):
        return NullArtist()

    def plot(self, *_args, **_kwargs):
        return (NullArtist(),)

    def scatter(self, *_args, **_kwargs):
        return NullArtist()

    def fill(self, *_args, **_kwargs):
        return (NullArtist(),)

    def text(self, *_args, **_kwargs):
        return NullArtist()

    def set_xlim(self, *_args, **_kwargs):
        return None

    def set_ylim(self, *_args, **_kwargs):
        return None

    def set_aspect(self, *_args, **_kwargs):
        return None

    def set_xlabel(self, *_args, **_kwargs):
        return None

    def set_ylabel(self, *_args, **_kwargs):
        return None

    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop
