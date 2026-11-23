"""Optional smoke tests: python test_app.py (requires the app dependencies)."""
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

app = AppTest.from_file(str(Path(__file__).with_name('app.py')), default_timeout=120).run()
assert not app.exception
app.sidebar.radio[0].set_value('Predict a score').run()
app.button[0].click().run()
assert not app.exception
score, model, frame = app.session_state['result']
assert np.isfinite(score) and frame.shape == (1, 19)
# Change a categorical input, then ensure the single-row prediction responds.
old_score = score
parent_select = next(x for x in app.selectbox if x.label == 'Parental Involvement')
parent_select.set_value('Low' if parent_select.value != 'Low' else 'High')
app.button[0].click().run()
assert not app.exception
assert not np.isclose(app.session_state['result'][0], old_score)
next(x for x in app.selectbox if x.label == 'Prediction model').set_value('Polynomial Regression')
app.button[0].click().run()
assert not app.exception and np.isfinite(app.session_state['result'][0])
app.sidebar.radio[0].set_value('Model lab').run()
assert not app.exception
app.sidebar.radio[0].set_value('Explore data').run()
app.slider[0].set_value((81, 81)).run()
assert not app.exception
assert len(app.info) > 0
print('PASS: all four pages, both predictors, categorical inference, empty filter.')
