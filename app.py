import streamlit as st

from calculation import prepare_data, calculate_daily_effort_matrix, calculate_overall_statistics, calculate_effort_distributions, calculate_quarter_hour_effort_matrix
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
st.header('General stats')
st.write(f"Total duration is {total_duration}. That equates to {total_duration.total_seconds() / 60 / 60 / ((52-6)*40):.2f} FTEs.")
st.write(f"Total effort (in childhours): {general_stats['total_effort']}")
st.write(f"Share of Parent D: {general_stats['total_share']*100:.1f}%")

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
