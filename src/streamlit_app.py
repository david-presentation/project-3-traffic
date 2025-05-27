import streamlit as st
from datetime import datetime
from datetime import timedelta

import pandas as pd
import geopandas as geopd
import numpy as np

# Reference to slider implementation can be found here
# https://docs.streamlit.io/develop/api-reference/widgets/st.slider

# Streamlit appears to refresh something in the background when the slider is moved, but it looks like I
# could use a session state to store the map and prevent it from going through it's initialisation cylce,
# because I noticed that if the map is void of any points, it redraws almost instantly without any
# sort of delay. So bascially, I need to store the map data to speed up the forced refresh.

# Reference use for implementation: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state

if 'node_data' not in st.session_state:
    date_range = pd.read_csv('../data/streamlit_nodes.csv', low_memory=False)
    st.session_state['node_data'] = date_range

if 'date_year' not in st.session_state:
    st.session_state['date_year'] = 2020

if 'date_month' not in st.session_state:
    st.session_state['date_month'] = 6

def change_date():
    del st.session_state['highways']
    st.session_state['date_year'] = start_time.year
    st.session_state['date_month'] = start_time.month

if 'filter_status' not in st.session_state:
    st.session_state['filter_status'] = True

if 'event_filter' not in st.session_state:
    st.session_state['event_filter'] = True


# Weight filtering
if 'light_filter' not in st.session_state:
    st.session_state['light_filter'] = False

# Weight filtering
if 'heavy_filter' not in st.session_state:
    st.session_state['heavy_filter'] = False

def change_heavy():
    if (st.session_state.heavy_filter):
        st.session_state.heavy_filter = False
    else:
        st.session_state.heavy_filter = True
    

def change_light():
    if (st.session_state.light_filter):
        st.session_state.light_filter = False
    else:
        st.session_state.light_filter = True


# Highway filtering
if 'highway_filter' not in st.session_state:
    st.session_state['highway_filter'] = ''


# Density colours
if 'density_colour' not in st.session_state:
    st.session_state['density_colour'] = False

# Row references:
# https://stackoverflow.com/questions/69492406/streamlit-how-to-display-buttons-in-a-single-line
# https://docs.streamlit.io/develop/api-reference/layout/st.columns
col1, col2, col3 = st.columns([1, 1, 3])

# Reference https://docs.streamlit.io/develop/api-reference/widgets/st.checkbox
with col1:
    # Reference https://docs.streamlit.io/develop/api-reference/widgets/st.toggle
    filtering = st.toggle("Apply filters")


if filtering:
    st.session_state.filter_status = False
else:
    st.session_state.filter_status = True

with col2:
    light_vehicles = st.checkbox("Light Vehicles",
    disabled=st.session_state.filter_status,
    value=st.session_state.light_filter,
    on_change=change_light)

with col3: 
    heavy_vehicles = st.checkbox("Heavy Vehicles",
    disabled=st.session_state.filter_status,
    value=st.session_state.heavy_filter,
    on_change=change_heavy)



# Reference https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox
highway_select = st.selectbox(
    "Filtered highway",
    ['', '10', '11', '12', '14', '15', '1N', '16', '18', '20', '20A', '22',
       '1B', '1C', '2', '21', '23', '24', '25', '26', '27', '28', '29',
       '3', '30', '31', '32', '39', '4', '41', '47', '5', '29A', '30A',
       '33', '34', '35', '36', '38', '50', '51', '43', '44', '45', '3N',
       '49', '54', '56', '57', '1K', '53', '58', '59', '1S', '6', '60',
       '63', '7', '73', '74', '74A', '75', '76', '77', '79', '8', '82',
       '65', '67', '69', '6A', '83', '85', '87', '88', '8A', '8B', '90',
       '93', '94', '95', '96', '99'],
    index=0,
    placeholder="Select a highway to filter",
    accept_new_options=False,
    disabled=st.session_state.filter_status
)

st.session_state['highway_filter'] = highway_select

events_data = { 
    "New Years Day" : { "day": 1, "month": 1, "year": 2024 },
    "Day After New Years Day" : { "day": 2, "month": 1, "year": 2024},
    "Waitangi" : { "day": 6, "month": 2, "year": 2024},
    "Good Friday" : { "day": 29, "month": 3, "year": 2024},
    "Easter Monday" : { "day": 1, "month": 4, "year": 2024},
    "Anzac Day" : { "day": 25, "month": 4, "year": 2024},
    "Kings Official Birthday" : { "day": 3, "month": 6, "year": 2024},
    "Matariki" : { "day": 28, "month": 6, "year": 2024},
    "Labour Day" : { "day": 28, "month": 10, "year": 2024},
    "Christmas Day" : { "day": 25, "month": 12, "year": 2024},
    "Boxing Day" : { "day": 26, "month": 12, "year": 2024},
    "Cyclone Gabrielle" : { "day": 5, "month": 2, "year": 2023},
    "Auckland Floods" : { "day": 27, "month": 1, "year": 2023},
    "Covid Lockdown" : { "day": 27, "month": 3, "year": 2020},
    "New Wellington Highway" : { "day": 30, "month": 3, "year": 2022},
    "Fieldays" : { "day": 12, "month": 6, "year": 2024},
    "FIFA Womens World Cup" : { "day": 20, "month": 6, "year": 2023}
}

event_col1, event_col2, event_col3 = st.columns([1, 1, 3])

toggle_col1, toggle_col2, toggle_col3, toggle_col4, toggle_col5, toggle_col6 = st.columns([4, 1, 1, 1, 1, 1])

with toggle_col1:
    # Reference https://docs.streamlit.io/develop/api-reference/widgets/st.toggle
    density_colour = st.toggle("Colours based on population density")

with toggle_col2:
    # Reference to badges found here: https://docs.streamlit.io/develop/api-reference/text/st.badge
    st.badge("Major", color='red')

with toggle_col3:
    st.badge("Large", color='orange')

with toggle_col4:
    st.badge("Medium", color='violet')

with toggle_col5:
    st.badge("Small", color='blue')

with toggle_col6:
    st.badge("Rural", color='green')


if density_colour:
    st.session_state.density_colour = True
else:
    st.session_state.density_colour = False

start_time = st.slider(
    "Traffic volume date",
    value=datetime(2020, 6, 1),
    format="MM/DD/YYYY",
    on_change=change_date,
    min_value=datetime(2018, 1, 1),
    max_value=datetime(2024, 1, 1),
    step=timedelta(days=29)
)

# It looks like Streamlit allows Folium maps to be dynamically updated,
# https://github.com/randyzwitch/streamlit-folium/pull/97

# Here are a few examples I've used to build our implementaiton:
# http://github.com/randyzwitch/streamlit-folium/blob/master/examples/pages/dynamic_updates.py
# https://github.com/randyzwitch/streamlit-folium/blob/master/examples/pages/dynamic_map_vs_rerender.py
# https://github.com/randyzwitch/streamlit-folium/blob/master/examples/pages/dynamic_layer_control.py

import folium
import folium.features
import geopandas as gpd
from streamlit_folium import st_folium


if 'highways' not in st.session_state:
    st.session_state['highways'] = []


highway_features = folium.FeatureGroup(name='highways')

for highway in st.session_state['highways']:
    highway_features.add_child(highway)


# Creating a colour mapping for the 'urban' column values. This will be used in the visualisation
colour_mapping = {
    'rural': 'green',
    'small': 'blue',
    'med': 'violet',
    'large': 'orange',
    'major': 'red'
}

if st.button('Reload Data'):
    del st.session_state['node_data']
    date_range = pd.read_csv('../data/streamlit_nodes.csv', low_memory=False)
    st.session_state['node_data'] = date_range


connections_map = folium.Map(location=(-41.15139566937802, 174.90173505272722), zoom_start=6.5, tiles='cartodb positron')

date_range = st.session_state.node_data
display_range = date_range[(date_range['year'] == st.session_state.date_year) & (date_range['month'] == st.session_state.date_month)]


if (st.session_state.filter_status == False):
    if (st.session_state.heavy_filter and st.session_state.light_filter):
        display_range = display_range[(display_range['classWeight'] == 'Heavy') | (display_range['classWeight'] == 'Light')]
    elif (st.session_state.heavy_filter):
        display_range = display_range[display_range['classWeight'] == 'Heavy']
    elif (st.session_state.light_filter):
        display_range = display_range[display_range['classWeight'] == 'Light']


if (st.session_state['highway_filter'] != '' and st.session_state.filter_status == False):
    display_range = display_range[display_range['sh'] == st.session_state['highway_filter']]



# Scaling values within dataset column, based on example provided here:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html 
from sklearn.preprocessing import MinMaxScaler


if (not display_range[['trafficCount']].shape[0] < 1):
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(display_range[['trafficCount']])
    display_range['trafficCount'] = min_max_scaler.transform(display_range[['trafficCount']])

# Used as a reference for generating colours
# https://python-visualization.github.io/folium/latest/advanced_guide/colormaps.html
import branca.colormap as cm
# linear_colour = cm.linear.YlOrRd_04.scale(0.00, 0.1)
linear_colour = cm.LinearColormap(['white', 'orange', 'red'], index=[0, 0.02, 0.5])

for i in range(0, display_range.shape[0]):
    node_index_value = display_range.index.values[i]

    if (st.session_state.density_colour):
        # Map the current nodes 'urban' value to a usable colour
        colour = display_range.loc[node_index_value, 'urban']
        colour = colour_mapping[colour]
    else:
        # Used as a reference for applying colours
        # https://python-visualization.github.io/folium/latest/advanced_guide/colormaps.html
        colour = linear_colour(display_range.loc[node_index_value, 'trafficCount'])

    # The geometry data points have been changed to strings, most likely due to exporting as a csv file.
    # To I'll have to split them up manually and cast them to float data types.
    x_coord = float(display_range.loc[node_index_value, 'geometry'].split(' ')[1][1:])
    y_coord = float(display_range.loc[node_index_value, 'geometry'].split(' ')[2][:-1])

    # It looks like the x and y coordinates are swapped around for some reason, but that shouldn't affect anything as long as we remember this.
    # There is also a popup property that I'll use for debugging to display the site reference, 'siteref', of each node
    site_reference = display_range.loc[node_index_value, 'siteref']

    # We could use the traffic volume associated with each site in order to influence the size of the circles drawn on the map
    weight_value = display_range.loc[node_index_value, 'trafficCount'] * 10
    
    circle = folium.Circle(location=[y_coord, x_coord], weight=8, color=colour, radius=weight_value, fill_color=colour, opacity=0.2)
    highway_features.add_child(circle)
    
    # Draw the connections between nodes
    # Implementation of checking for nan values from here: https://numpy.org/doc/2.1/reference/generated/numpy.isnan.html
    if (not np.isnan(display_range.loc[node_index_value, 'neighour_next'])):
        # Get the index of the next node
        next_index = display_range.loc[node_index_value, 'neighour_next']
        
        next_x_coord = float(display_range.loc[next_index, 'geometry'].split(' ')[1][1:])
        next_y_coord = float(display_range.loc[next_index, 'geometry'].split(' ')[2][:-1])

        weight_value = display_range.loc[node_index_value, 'trafficCount'] * 10
        
        if (st.session_state.density_colour):
            line = folium.PolyLine([(y_coord, x_coord), (next_y_coord, next_x_coord)], weight=weight_value, color=colour,)
        else:
            colour = linear_colour(display_range.loc[next_index, 'trafficCount'])
            line = folium.PolyLine([(y_coord, x_coord), (next_y_coord, next_x_coord)], weight=weight_value * 2, color=colour,)
        highway_features.add_child(line)

st_folium(connections_map, feature_group_to_add=highway_features, height=900, width=800)
