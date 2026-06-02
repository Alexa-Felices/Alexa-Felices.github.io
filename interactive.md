```python
# Import necessary libraries
import altair as alt
import pandas as pd

# Load in data from local source
df = pd.read_csv('data/renewable_energy_share_2000_2025.csv')

# Remove aggregates (iso code = NA)
df = df.dropna(subset=['iso_code'])

# Reformat year and remove incomplete data (2025)
df['Year'] = pd.to_datetime(df['year'], format='%Y').dt.year
df = df[df['Year'] != 2025]

# Rename columns for nicer axis labels
df = df.rename(columns={'country': 'Country', 'population': 'Population', 'solar_electricity': 'Solar',
                        'wind_electricity': 'Wind', 'hydro_electricity': 'Hydro', 'nuclear_electricity': 'Nuclear',
                        'coal_electricity': 'Coal', 'gas_electricity': 'Gas', 'oil_electricity': 'Oil',
                        'biofuel_electricity': 'Biofuel', 'other_renewable_electricity': 'Other Renewable'})

# Get dataframe of total energy usage and replace NA electricity values with 0
df_total_energy = df[['Country', 'Year', 'Population', 'Solar', 'Wind', 'Hydro', 'Nuclear', 'Coal', 'Gas', 'Oil',
                      'Biofuel', 'Other Renewable']]

df_total_energy = df_total_energy.fillna(0)

# Change population to millions
df_total_energy['Population'] = df_total_energy['Population']/1000000

# Pivot chart for easier plotting
df_total_energy_long = df_total_energy.melt(
    id_vars=['Year', 'Country', 'Population'],
    value_vars=['Solar', 'Wind', 'Hydro', 'Nuclear', 'Coal', 'Gas', 'Oil', 'Biofuel', 'Other Renewable'],
    var_name='Energy Type',
    value_name='TWh'
)

df_total_energy_long = df_total_energy_long[df_total_energy_long['TWh'] >= 0]

# Implement dropdown
dropdown = alt.binding_select(options=df_total_energy['Country'].unique(), name="Select a Country: ")

# Implement selection
selection = alt.selection_point(fields=['Country'], bind=dropdown)

# Generate bar chart with selection
bar1 = alt.Chart(df_total_energy_long[df_total_energy_long['Year'] == 2024]).mark_bar().encode(
    x='Energy Type:O',
    y=alt.Y('sum(TWh):Q', axis=alt.Axis(title='2024 Production Type (TWh)')).scale(domainMin=0),
    color=alt.Color('Energy Type:N', legend=alt.Legend(orient='left'), scale=alt.Scale(scheme='viridis')),
    tooltip=['Energy Type', "sum(TWh)"],
    opacity=alt.condition(selection, alt.value(1), alt.value(2))
).add_params(selection).transform_filter(selection)

# Generate bar chart of total by year
bar2 = alt.Chart(df_total_energy_long).mark_bar(color='#479ADE').encode(
    x='Year:O',
    y=alt.Y('sum(TWh):Q', axis=alt.Axis(title='Total Energy Production (TWh)', titleColor='#479ADE')),
    tooltip=["Year", "sum(TWh)"],
    opacity=alt.condition(selection, alt.value(1), alt.value(2))
).add_params(selection).transform_filter(selection)

# Generate line graph of population by year
line = bar2.mark_line(stroke='#DE8B47', interpolate='monotone').encode(
    x='Year:O',
    y=alt.Y('mean(Population):Q').axis(title='Population (Millions)', titleColor='#DE8B47')
).add_params(selection).transform_filter(selection)

# Add line graph to second bar chart
bar2 = alt.layer(bar2, line).resolve_scale(
    y='independent'
)

# Combine and format charts
combined_charts = (bar1 | bar2).properties(
    title=alt.TitleParams('Energy Production by Country and Type from 2000-2024', anchor='middle', fontSize=35)
).configure_axis(
    labelFontSize=12, titleFontSize=20
).configure_view(
    strokeWidth=0
).properties(
    padding={"left": 40, "right": 40, "top": 30, "bottom": 30}
)

# Save for later use
combined_charts.save('country_energy_split.html')
```
