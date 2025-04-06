import pandas as pd

from config import DEFAULT_FACTOR, PARENTS, EFFORT_FACTORS, DEFAULT_FILENAME

S_PER_H = 60*60


def prepare_data():
    df = load_data()
    df = calculate_datetime_values(df)
    df = calculate_factor(df)
    df = calculate_share(df)
    df = calculate_effort(df)
    return df


def load_data(filename=DEFAULT_FILENAME):
    df = pd.read_csv(filename, sep=';')
    df['temp'] = df.name.str.split(':')
    df['responsible'] = df['temp'].str[0].str.strip()
    df['children'] = df['temp'].str[1].str.strip()
    df = df.drop(['#', 'name', 'calendar', 'comment', 'temp'], axis=1)

    # Correct missspellings
    df.loc[df.responsible == 'Beide', 'responsible'] = 'Geteilt'
    df.loc[df.responsible == 'Gestellt', 'responsible'] = 'Geteilt'

    # Convert to datetimes
    df.start = pd.to_datetime(df.start, utc=True)
    df.end = pd.to_datetime(df.end, utc=True)
    # df['duration'] = pd.to_timedelta(df.duration)
    df.duration = df.end - df.start

    return df

def calculate_overlap(df):
    # Calculate overlap
    overlap = (df['start'] < df['end'].shift(-1)) & (df['end'] > df['start'].shift(-1))
    overlapping_events = df[overlap]
    return df

def calculate_datetime_values(df):
    # Calculate values for datetime analysis
    df['start_quarter'] = quarter_hours(df['start'])
    df['end_quarter'] = quarter_hours(df['end'])
    df['date'] = df['start'].dt.date
    df['month'] = df['start'].dt.month
    df['weekday'] = df['start'].dt.weekday
    df['year'] = df['start'].dt.year
    return df

def quarter_hours(dt_series: pd.Series):
    return dt_series.dt.hour * 4 + dt_series.dt.minute // 15

def calculate_factor(df):
    df['factor'] = DEFAULT_FACTOR
    for child_name, factor in EFFORT_FACTORS.items():
        df.loc[df.children == child_name, 'factor'] = factor
    
    return df

def calculate_share(df):
    equal_share = 1 / len(PARENTS)
    for i, parent in enumerate(PARENTS):
        col_name = 'share_' + str(i)
        df[col_name] = 0.0
        df.loc[df.responsible == parent, col_name] = 1.0
        df.loc[df.responsible == 'Geteilt', col_name] = equal_share
    
    return df

def calculate_effort(df):
    # Calculate overall statistics
    df['effort'] = (df['duration'].dt.total_seconds() / S_PER_H) * df['factor']
    return df


def calculate_overall_statistics(df):
    # Calculate total share per parent
    total_duration = df['duration'].sum()
    duration_by_responsible = df.groupby('responsible')['duration'].sum().dt.total_seconds() / S_PER_H
    max_responsible = duration_by_responsible.iloc[duration_by_responsible.argmax()]
    min_responsible = duration_by_responsible.iloc[duration_by_responsible.argmin()]
    shared_responsible = total_duration.total_seconds() / S_PER_H - max_responsible - min_responsible
    diff_as_percent_of_min_responsible_share = (max_responsible + shared_responsible) / (min_responsible + shared_responsible) - 1
    total_effort = df['effort'].sum()
    total_share = (df['effort'] * df['share_0']).sum() / total_effort
    return {
        'total_duration': total_duration,
        'total_effort': total_effort,
        'total_share': total_share,
        'difference_between_parents': max_responsible - min_responsible
    }

def calculate_effort_distributions(df):
    # Calculate trend over year
    monthly_total_efforts = df.groupby(['month'])['effort'].sum() 
    monthly_effort_shares = (df['effort'] * df['share_0']).groupby(df['month']).sum() / monthly_total_efforts

    # Calculate distribution over weekdays
    total_efforts_per_weekday = df.groupby(['weekday'])['effort'].sum()
    effort_shares_per_weekday = (df['effort'] * df['share_0']).groupby(df['weekday']).sum() / total_efforts_per_weekday
    # to normalize (divide by num. occurrences of that weekday)
    total_efforts_per_weekday = total_efforts_per_weekday / len(df.date.unique()) * 7 
    return {
        'monthly_total_efforts': monthly_total_efforts,
        'monthly_effort_shares': monthly_effort_shares,
        'total_efforts_per_weekday': total_efforts_per_weekday,
        'effort_shares_per_weekday': effort_shares_per_weekday
    }


def calculate_daily_effort_matrix(df: pd.DataFrame):
    # Calculate heatmap for year
    daily_total_efforts = df.groupby(['date'])['effort'].sum() 
    daily_effort_df = (
        (df['effort'] * df['share_0'])
        .groupby(df['date'])
        .sum() / daily_total_efforts
    ).to_frame(name='share')
    daily_effort_df['intensity'] = daily_total_efforts / daily_total_efforts.max()
    daily_effort_df['weekday'] = pd.to_datetime(daily_effort_df.index).weekday
    daily_effort_df['month'] = pd.to_datetime(daily_effort_df.index).month
    daily_effort_df['year'] = pd.to_datetime(daily_effort_df.index).year
    daily_effort_df['week'] = pd.to_datetime(daily_effort_df.index).isocalendar().week
    # Deal with problems with non-unique index 
    min_year = daily_effort_df['year'].min()
    max_week = daily_effort_df['week'].max()
    daily_effort_df['week'] += max_week * (daily_effort_df['year'] - min_year)
    # Deal with special case of new year's first week
    first_week_new_year = (daily_effort_df['month'] == 12) & (daily_effort_df['week'] == 1)
    daily_effort_df.loc[first_week_new_year, 'week'] += max_week

    effort_matrix = daily_effort_df.pivot(index='week', columns='weekday', values=['share', 'intensity']).fillna(0)
    return effort_matrix

# Calculate child-free time per parent

# Calculate distribution over daytime
def calculate_quarter_hour_effort_matrix(df: pd.DataFrame):
    weekday_index = pd.Index(range(7), dtype='int32', name='weekday')
    quarter_hour_efforts = pd.DataFrame(index=weekday_index.copy())
    quarter_hour_shares = pd.DataFrame(index=weekday_index.copy())
    min_quarter = df['start_quarter'].min()
    max_quarter = df['end_quarter'].max()
    for quarter in range(min_quarter, max_quarter):
        matches = df.loc[(df['start_quarter'] <= quarter) & (quarter < df['end_quarter']), ['weekday', 'share_0', 'factor']]
        effort = matches.groupby('weekday')['factor'].sum()
        share = (matches['share_0'] * matches['factor']).groupby(matches['weekday']).sum() / effort
        quarter_hour_efforts[quarter] = effort
        quarter_hour_shares[quarter] = share

    num_weeks = (df.start.max() - df.start.min()).days // 7
    max_factor = 2
    normed_quarter_hour_efforts = quarter_hour_efforts / num_weeks / max_factor
    return normed_quarter_hour_efforts.fillna(0), quarter_hour_shares.fillna(0)


def date_from_day_of_year(day_of_year: int, year: int = datetime.now().year):
    result_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    return result_date


# fig = plot_heatmap(quarter_hour_efforts.fillna(0))
# fig.show()
# fig = plot_heatmap(quarter_hour_shares, intensity_matrix=normed_quarter_hour_efforts)
# fig.show()