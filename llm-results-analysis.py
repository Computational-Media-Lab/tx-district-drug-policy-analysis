import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Read in all results files and create concatenated dataframes

folders = ['DISTRICTS']

df_summary = pd.concat([pd.read_csv(f'results-v2/{f}/{f}_summary.csv') for f in folders], ignore_index=True)

df_detailed = pd.concat(
    [pd.read_csv(f'results-v2/{f}/{f}_detailed.csv').assign(district=f) for f in folders],
    ignore_index=True)

# CHECK: district names must match between the two
print(set(df_summary['district']) ^ set(df_detailed['district']))  # should print set()
print(df_detailed['Topic Category'].unique())  # use to fill category_map


# Deduplicate texts within each district and category
df_detailed['Text_norm'] = (df_detailed['Text'].str.lower()
                            .str.replace(r'\s+', ' ', regex=True).str.strip())

df_dedup = (df_detailed
            .drop_duplicates(subset=['district', 'Topic Category', 'Text_norm'])
            .drop(columns='Text_norm')
            .reset_index(drop=True))

print(len(df_detailed), '->', len(df_dedup))


# Update the summary
# First look at what 'Topic Category' contains & build mapping to the summary column names
category_map = {
     'Disciplinary Other': 'number of times any other terms are mentioned that relate to disciplinary approaches to student drug and alcohol use',
     'Expulsion': 'number of times expulsion or synonyms are mentioned',
     'Disciplinary School': 'number of time transfer or referral to disciplinary alternative school is mentioned', 
     'Prevention Programs': 'number of times drug and alcohol use prevention programs are mentioned', 
     'Recovery School': 'number of times transfer or referral to recovery schools are mentioned', 
     'Restorative Justice': 'number of times restorative justice approaches or synonyms are mentioned', 
     'Juvenile Justice Program': 'number of times transfer or referral to juvenile justice program is mentioned',
     'Health Staff': 'number of times health-services staff such as school nurses, social workers, psychologists, or substance use counselors are mentioned', 
     'Health Other': 'number of times any other terms are mentioned that relate to a health-oriented approach to student drug and alcohol use', 
     'Suspension': 'number of times suspension or synonyms are mentioned', 
     'Police': 'number of times police or school resource officers are mentioned', 
     'Trauma Care': 'number of times trauma-informed care or synonyms are mentioned', 
     'Arrest': 'number of times arrest or synonyms are mentioned', 
     'Treatment Program': 'number of times transfer or referral to treatment programs are mentioned'
}

# Recount per district x category
counts = (df_dedup.groupby(['district', 'Topic Category'])
                  .size().unstack(fill_value=0)
                  .rename(columns=category_map))

# Update summary (district stays a column)
df_summary_old = df_summary.copy()
cols = [c for c in counts.columns if c in df_summary.columns]
df_summary[cols] = (counts[cols].reindex(df_summary['district'])
                    .fillna(0).astype(int).to_numpy())

count_cols = [c for c in df_summary.columns if c.startswith('number of time')]
df_summary['total number of drug or alcohol terms mentioned'] = df_summary[count_cols].sum(axis=1)

# Compare old vs new
print(df_summary_old.set_index('district')[count_cols] - df_summary.set_index('district')[count_cols])

# Overwrite detailed
df_detailed = df_dedup


#===============================================#
#           SUMMARY RESULTS ANALYSIS            #
#===============================================#

# Create two new summary columns for disciplinary & health/treatment terms mentioned
df_summary['total number of disciplinary terms mentioned'] = (
    df_summary['number of times suspension or synonyms are mentioned'] +
    df_summary['number of times expulsion or synonyms is mentioned'] +
    df_summary['number of time transfer or referral to disciplinary alternative school is mentioned'] +
    df_summary['number of times transfer or referral to juvenile justice program is mentioned'] + 
    df_summary['number of times arrest or synonyms are mentioned'] + 
    df_summary['number of times drug sniffing dogs or other drug screening procedures are mentioned'] +
    df_summary['number of times police or school resource officers are mentioned'] +
    df_summary['number of times any other terms are mentioned that relate to disciplinary approaches to student drug and alcohol use']
)

df_summary['total number of health or treatment terms mentioned'] = (
    df_summary['number of times transfer or referral to treatment programs are mentioned'] +
    df_summary['number of times transfer or referral to recovery schools are mentioned'] +
    df_summary['number of times school based health centers or synonyms are mentioned'] + 
    df_summary['number of times transfer or referral to mental health clinics or other off campus health facilities are mentioned'] + 
    df_summary['number of times trauma-informed care or synonyms are mentioned'] + 
    df_summary['number of times restorative justice approaches or synonyms are mentioned'] + 
    df_summary['number of times drug and alcohol use prevention programs are mentioned'] + 
    df_summary['number of times health-services staff such as school nurses, social workers, psychologists, or substance use counselors are mentioned'] +
    df_summary['number of times any other terms are mentioned that relate to a health-oriented approach to student drug and alcohol use']
)

col1 = df_summary.pop("total number of disciplinary terms mentioned")
col2 = df_summary.pop("total number of health or treatment terms mentioned")

df_summary.insert(3, col1.name, col1)
df_summary.insert(4, col2.name, col2)


df_summary['d_ratio'] = df_summary['total number of disciplinary terms mentioned'] / (df_summary['total number of disciplinary terms mentioned'] + df_summary['total number of health or treatment terms mentioned'])
df_summary['h_ratio'] = df_summary['total number of health or treatment terms mentioned'] / (df_summary['total number of disciplinary terms mentioned'] + df_summary['total number of health or treatment terms mentioned'])

# Create alias column for districts and drop original district column name
district_type_map = {"District Names": "District Aliases"} # Redacted for data privacy purposes

df_summary['district-alias'] = df_summary['district'].map(district_type_map)
df_summary = df_summary.drop(columns = ['district', 'state'])

df_d_ratio = df_summary.sort_values(by = 'd_ratio').reset_index(drop = True)

fig, ax = plt.subplots(figsize = (12, 5), dpi = 150)

ax.axhline(0, color = '#cccccc', linestyle = '--', linewidth = 1.5, zorder = 1)
ax.scatter(df_summary['d_ratio'], np.zeros(len(df_summary)), color = '#2b5c8f', s = 120, zorder = 3)

offsets = [20, -30, 45, -65, 85, -30, 35, -45, 70, 50, 35, -45]

for i, row in df_summary.iterrows():
    offset = offsets[i % len(offsets)]
    ax.annotate(
        row['district-alias'],
        (row['d_ratio'], 0),
        xytext=(0, offset),
        textcoords="offset points",
        ha='center',
        va='bottom' if offset > 0 else 'top',
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8)
    )

ax.set_ylim(-0.1, 0.1)
ax.get_yaxis().set_visible(False) 
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Disciplinary Terms Ratio', fontsize = 12, labelpad = 10)
ax.set_title('District Disciplinary Terms Index Spectrum', fontsize = 14, pad = 10)
plt.tight_layout()
plt.show()


df_d_ratio = df_summary.sort_values(by = 'h_ratio').reset_index(drop = True)

fig, ax = plt.subplots(figsize = (12, 5), dpi = 150)

ax.axhline(0, color = '#cccccc', linestyle = '--', linewidth = 1.5, zorder = 1)
ax.scatter(df_summary['h_ratio'], np.zeros(len(df_summary)), color = '#2b5c8f', s = 120, zorder = 3)

offsets = [20, -30, 45, -65, 85, -30, 35, -45, 70, 50, 35, -45]

for i, row in df_summary.iterrows():
    offset = offsets[i % len(offsets)]
    ax.annotate(
        row['district-alias'],
        (row['h_ratio'], 0),
        xytext=(0, offset),
        textcoords="offset points",
        ha='center',
        va='bottom' if offset > 0 else 'top',
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8)
    )

ax.set_ylim(-0.1, 0.1)
ax.get_yaxis().set_visible(False) 
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Health Terms Ratio', fontsize = 12, labelpad = 10)
ax.set_title('District Health Terms Index Spectrum', fontsize = 14, pad = 10)
plt.tight_layout()
plt.show()


#===============================================#
#           DETAILED RESULTS ANALYSIS           #
#===============================================#
df_detailed.info()

df_detailed.head(10)


# Create new column called PDF-Category extracting keyword categories from PDF column
categories = ['LEGAL', 'LOCAL', 'REGULATION', 'EXHIBIT']

df_detailed['PDF-Category'] = df_detailed['PDF'].str.extract(r'(' + '|'.join(categories) + r')')


# Create new column Topic-Category-Condensed based on Topic-Category value
disciplinary_sentences = [
    'number of times suspension or synonyms are mentioned',
    'number of times expulsion or synonyms is mentioned', 
    'number of time transfer or referral to disciplinary alternative school is mentioned', 
    'number of times transfer or referral to juvenile justice program is mentioned',  
    'number of times arrest or synonyms are mentioned', 
    'number of times drug sniffing dogs or other drug screening procedures are mentioned', 
    'number of times police or school resource officers are mentioned', 
    'number of times any other terms are mentioned that relate to disciplinary approaches to student drug and alcohol use'
]

# Filter out 'Total Number of drug or alcohol terms mentioned' column to prevent overcount
df_detailed = df_detailed.loc[df_detailed['Topic Category'] != 'total number of drug or alcohol terms mentioned']

df_detailed['Topic-Category-Condensed'] = np.where(df_detailed['Topic Category'].isin(disciplinary_sentences), 'Disciplinary', 'Health')


df_detailed.tail(20)


# Write out detailed results
df_detailed.to_csv('results-v2/overall-detailed.csv', index=False)


# Create table showing value counts of condensed topic category for each PDF category
count_table = pd.crosstab(df_detailed['PDF-Category'], df_detailed['Topic-Category-Condensed'])
display(count_table)


ax = count_table.plot(kind = 'bar')

# Add count labels to each bar
for container in ax.containers:
    ax.bar_label(container, fmt = "%d", padding = 3)

ax.set_ylim(top=ax.get_ylim()[1] * 1.10)

plt.xlabel('PDF Category')
plt.ylabel('Count')
plt.title('Disciplinary & Health Counts for Each PDF Category')
plt.xticks(rotation = 0)
plt.legend(title='Topic-Category-Condensed')
plt.tight_layout()
plt.show()