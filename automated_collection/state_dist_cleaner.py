import pandas as pd
from geopy.distance import great_circle

# https://github.com/jasperdebie/VisInfo/blob/master/us-state-capitals.csv
caps = pd.read_csv("collected_data/us_state_capitals.csv", index_col = 0)

# https://data.world/bryon/state-adjacency
adjs = pd.read_csv("collected_data/state_adjacency.csv")


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

abbrev = dict(map(reversed, us_state_abbrev.items()))

c_adj = []
for ind in adjs.index:
    state = abbrev[adjs.loc[ind,"STATE"]]
    adjer = abbrev[adjs.loc[ind, "ADJ"]]

    state_geo = (float(caps.loc[caps.name == state].latitude), float(caps.loc[caps.name == state].longitude))
    adjer_geo = (float(caps.loc[caps.name == adjer].latitude), float(caps.loc[caps.name == adjer].longitude))
    dist = great_circle(state_geo, adjer_geo).miles

    c_adj.append([state, adjer, dist])


us_adj = pd.DataFrame(c_adj, columns = ["state", "adj", "distance"])
us_adj.to_csv("collected_data/state_adjacency.csv")


