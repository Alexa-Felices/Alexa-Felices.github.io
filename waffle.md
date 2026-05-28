```python
# Necessary imports for this chart
import pandas as pd
import matplotlib.pyplot as plt
from pywaffle import Waffle
from pypalettes import load_palette
from pyfonts import load_google_font
from drawarrow import fig_arrow
from highlight_text import fig_text

# Read in data from local storage
df = pd.read_csv('data/renewable_energy_share_2000_2025.csv')

# Remove 2025 data, convert year to datetime
df['Year'] = pd.to_datetime(df['year'], format='%Y').dt.year
df = df[df['Year'] != 2025]

# Grab relevant energy production columns and year
df_total_energy = df[['Year', 'solar_electricity', 'wind_electricity', 'hydro_electricity',
                      'nuclear_electricity', 'coal_electricity', 'gas_electricity', 'oil_electricity',
                      'biofuel_electricity', 'other_renewable_electricity']]

# Drop NA values
df_total_energy = df_total_energy.dropna()

# Pivot data so that the types of energy production are converted to labels with TWh as their value
df_total_energy_long = df_total_energy.melt(
    id_vars=['Year'],
    value_vars=['solar_electricity', 'wind_electricity', 'hydro_electricity', 'nuclear_electricity',
                'coal_electricity', 'gas_electricity', 'oil_electricity', 'biofuel_electricity',
                'other_renewable_electricity'],
    var_name='Energy Type',
    value_name='TWh'
)

# Get all unique labels from the data for indexing the dataframe
years_all = df_total_energy_long['Year'].unique()
energy_all = df_total_energy_long['Energy Type'].unique()
df_total_energy_final = df_total_energy_long.groupby(['Year', 'Energy Type'], as_index=False).sum()
idx = pd.MultiIndex.from_product(
    [years_all, energy_all], names=['Year', 'Energy Type']
)

df_total_energy_final = (df_total_energy_final.set_index(['Year', 'Energy Type'])
                         .reindex(idx, fill_value=0).reset_index())
df_total_energy_final = df_total_energy_final.sort_values(['Year', 'Energy Type'])

# Label renewable energy for each year
df_total_energy_final['is_renewable'] = df_total_energy_final['Energy Type'].isin(['solar_electricity',
                                                                                   'wind_electricity',
                                                                                   'hydro_electricity',
                                                                                   'biofuel_electricity',
                                                                                   'other_renewable_electricity'])

# Fonts used in the chart
lfont = load_google_font('Montserrat', weight="light")
regfont = load_google_font('Montserrat')
italfont = load_google_font('Montserrat', weight="bold", italic=True)

# Create color palette (manually set renewable sources to green hues)
n_colors = df_total_energy_final['Energy Type'].nunique()
pal= load_palette('Ofrenda')
colors = [pal[0]] + [pal[3]] + [pal[8]] + [pal[2]] + [pal[6]] + [pal[9]] + [pal[7]] + [pal[1]] + [pal[6]] + ['white']

# Get sum of all TWh produced in 2024 for building the waffle chart
max_year_value = df_total_energy_final.groupby('Year')['TWh'].agg('sum').max()

# Initialize figure
n_cols = df_total_energy_final['Year'].nunique()
fig, axs = plt.subplots(ncols=n_cols, figsize=(12, 8))

# Add y scale (pywaffle does not build it automatically)
y_ticks = [0, 30000, 60000, 90000, 120000, 150000, 180000, 210000]
for y_tick in y_ticks:
    axs[0].text(
        x=-0.25,
        y=y_tick / max_year_value,
        s=f"{y_tick}"[:-4]+"0K",
        size=10,
        va="center",
        ha="right",
        transform=axs[0].transAxes,
    )
    axs[0].axhline(
        y=y_tick / max_year_value,
        xmin=-0.2,
        xmax=-0.3,
        clip_on=False,
        color="black",
        linewidth=0.7,
    )
axs[0].axvline(x=-0.03, ymax=1.05, clip_on=False, color="black", linewidth=0.7)

# Build waffle chart 
for year, ax in zip(df_total_energy_final['Year'].unique(), axs):
    values = list(df_total_energy_final[df_total_energy_final['Year'] == year]['TWh'].values)
    values = sorted(values, reverse=True)
    values.append(max_year_value - sum(values))

    Waffle.make_waffle(
        ax=ax,
        rows=50,
        columns=10,
        values=values,
        vertical=True,
        colors=colors,
    )

    ax.text(x=0.1, y=-0.04, s=f"{int(year)}", fontsize=9, ha='center', font=regfont)
    adj_y = 0.03
    if year == 2000 or year == 2024:
        ax.text(
            x=0.099,
            y=sum(values[:-1]) / max_year_value + adj_y,
            s=f"{sum(values[:-1]):3g}"[:-3]+"K",
            fontsize=12,
            ha="center",
            font=italfont,
        )

# Create labels for categories
label_params = dict(
    x=1.05,
    size=15,
    transform=axs[n_cols - 1].transAxes,
    font=regfont,
    clip_on=False,
)
axs[n_cols - 1].text(y=0.15, s="Coal", color=colors[0], **label_params)
axs[n_cols - 1].text(y=0.40, s="Gas", color=colors[1], **label_params)
axs[n_cols - 1].text(y=0.58, s="Hydro", color=colors[2], **label_params)
axs[n_cols - 1].text(y=0.70, s="Nuclear", color=colors[3], **label_params)
axs[n_cols - 1].text(y=0.80, s="Wind", color=colors[4], **label_params)
axs[n_cols - 1].text(y=0.88, s="Solar", color=colors[5], **label_params)
axs[n_cols - 1].text(y=0.94, s="Oil", color=colors[7], **label_params)
axs[n_cols - 1].text(y=0.98, s="Biofuel", color=colors[8], **label_params)

fig.text(
    x=0.65,
    y=0.88,
    s="Other Renewable",
    size=15,
    font=regfont,
    color=colors[6],
)

arrow_params = dict(
    ax=axs[n_cols - 1],
    transform=axs[n_cols - 1].transAxes,
    clip_on=False,
)

fig_arrow(
    [0.81, 0.89], [0.88, 0.85], color=colors[6], radius=-0.3, clip_on=False, fig=fig
)

# Title of chart
title = """
Worldwide Energy Production from 2000-2024 \nby Source and in TWh
"""
fig.text(x=0.14, y=0.9, s=title, size=20, font=regfont, va="top")

# Get percentage of renewable energy production in 2000
total_2000 = df_total_energy_final[df_total_energy_final['Year'] == 2000]['TWh'].sum()
ren_energy_2000 = df_total_energy_final[df_total_energy_final['Year'] == 2000]
ren_energy_2000 = ren_energy_2000[ren_energy_2000['is_renewable'] == True]
perc_ren_energy_2000 = ren_energy_2000['TWh'].sum() / total_2000 * 100

# Get percentage of renewable energy production in 2024
total_2024 = df_total_energy_final[df_total_energy_final['Year'] == 2024]['TWh'].sum()
ren_energy_2024 = df_total_energy_final[df_total_energy_final['Year'] == 2024]
ren_energy_2024 = ren_energy_2024[ren_energy_2024['is_renewable'] == True]
perc_ren_energy_2024 = ren_energy_2024['TWh'].sum() / total_2024 * 100

# Report total renewable energy production in 2024
subtitle = f"""
<Renewable energy sources> represent \n<{perc_ren_energy_2024:.0f}%> of all energy produced in 2024.
"""
fig_text(
    x=0.14,
    y=0.75,
    s=subtitle,
    size=15,
    font=italfont,
    color="#afafaf",
    va="top",
    highlight_textprops=[{"color": colors[6]}, {"color": colors[6]}],
)

# Data reference labels
source_params = dict(ha="right", font=lfont, color="#9a9a9a", size=10)
fig.text(x=0.9, y=0.04, s="Data retrieved: 2026-05-24", **source_params)
fig.text(x=0.9, y=0.02, s="Viz: Alexa Felices", **source_params)

# Show plot
plt.show()
```
