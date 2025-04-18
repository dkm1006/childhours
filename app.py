import streamlit as st

from calculation import prepare_data, calculate_daily_effort_matrix, calculate_overall_statistics, calculate_effort_distributions, calculate_quarter_hour_effort_matrix, date_from_day_of_year
from plotting import plot_heatmap_seaborn, plot_heatmap_plotly


prepare_data_cached = st.cache_data(prepare_data)

st.title('A Year of Childcare')

data_load_state = st.text('Preparing data...')
df = prepare_data_cached()
data_load_state.text('Preparing data...done!')
st.header('Raw data')
st.write(df)

general_stats = calculate_overall_statistics(df)
total_duration = general_stats['total_duration']
total_hours = total_duration.total_seconds() / 60 / 60
total_spanned_period = (df.start.max() - df.start.min()).days
avg_hours_per_day = total_hours / total_spanned_period
difference_between_parents_in_typical_days_per_year = general_stats['difference_between_parents'] / avg_hours_per_day / total_spanned_period * 365

st.header('General stats')
st.write(f"Total duration of childcare is {total_duration} or {total_hours} hours over {total_spanned_period} days.")
st.write(f"It follows that the total duration equates to {total_hours / ((52-6)*40)/ total_spanned_period * 365:.2f} FTEs.")
st.write(f"Total effort (in childhours, i.e. duration multiplied by number of children to care for): {general_stats['total_effort']}")
st.write(f"Share of Parent D: {general_stats['total_share']*100:.1f}%")
st.write(f"Difference between parents: {general_stats['difference_between_parents']} hours")
st.write(f"On average that means {general_stats['difference_between_parents'] / total_spanned_period * 60:.0f} extra minutes per day or {general_stats['difference_between_parents'] / total_spanned_period * 7:.0f} extra hours per week.")
st.write(f"Parent D would solely take care of the kinds until day {int(difference_between_parents_in_typical_days_per_year)}, i.e. {date_from_day_of_year(int(difference_between_parents_in_typical_days_per_year)).strftime('%-d %B')}, if the rest of the year would be shared equally.")


st.header('Trends')
trends = calculate_effort_distributions(df)

st.subheader('Monthly efforts')
st.line_chart(trends['monthly_total_efforts'], x_label='Month', y_label='Effort [h]')
st.line_chart(trends['monthly_effort_shares'], x_label='Month', y_label='Share')

st.subheader('Efforts per weekday')
st.bar_chart(trends['total_efforts_per_weekday'], x_label='Weekday', y_label='Effort [h]')

effort_matrix = calculate_daily_effort_matrix(df)
st.subheader('Effort distribution')
# st.write(effort_matrix)

fig = plot_heatmap_seaborn(effort_matrix['share'].T, effort_matrix['intensity'].T, title='Work share over the year', cbar_kws={"label": "Work Share", 'shrink': 0.5})
st.write(fig)
fig = plot_heatmap_seaborn(effort_matrix['intensity'].T, title='Effort intensity over the year', colourmap='inferno', cbar_kws={"label": "Intensity", 'shrink': 0.5})
st.write(fig)

quarter_hour_efforts, quarter_hour_shares = calculate_quarter_hour_effort_matrix(df)
fig = plot_heatmap_seaborn(quarter_hour_shares, quarter_hour_efforts, title='Share of work over the day', cbar_kws={"label": "Work Share", 'shrink': 0.5})
st.write(fig)
fig = plot_heatmap_plotly(quarter_hour_efforts)
st.write(fig)
