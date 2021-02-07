from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
from wtforms.widgets import TextArea


class SummaryForm(FlaskForm):
    review_text = StringField(u'Review text', widget=TextArea(), validators=[DataRequired(), Length(10, 300)])
    submit = SubmitField('Summarize')
