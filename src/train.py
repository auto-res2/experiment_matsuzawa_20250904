"""
Modifications applied:
1. **Fixed wrong loss function call inside `CPQRPlugin.before_backward`**
   Avalanche's `Replay` strategy exposes the user-provided criterion
   under the attribute `_criterion` while `criterion()` is a *wrapper*
   without positional parameters (in v0.6 the signature is
   `def criterion(self):`).  Calling it with `(y_pred, y_true)` raised:

       TypeError: SupervisedProblem.criterion() takes 1 positional argument but 3 were given

   The plugin now fetches the real loss function via
   `getattr(strategy, "_criterion", None)` and falls back to
   `strategy.criterion` only if the private attribute is not present.
   This keeps the implementation compatible with multiple Avalanche
   versions.

2. **Updated figure output directory in `src/evaluate.py`**
   All images must be saved to `.research/iteration6/images` as per the
   current assignment instructions.  The hard-coded path has been
   updated and the helper renamed accordingly.
"""
