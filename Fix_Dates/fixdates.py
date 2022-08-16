import pandas
import numpy
from datetime import datetime

# Read the .csv file and parse it into a DataFrame
df = pandas.read_csv('Peru_2019_AudioMoth_Data.csv')

good_AM = [ 
        "AM-1", "AM-2", "AM-22", "AM-24",
        "AM-25", "AM-26", "AM-27", "AM-29",
        "AM-30", "WWF-1",  "WWF-3", "WWF-4", 
        "WWF-5", "AM-18", "AM-12", "AM-13"
    ]
add = 0
for i in range(len(df)):
    if df.loc[i, "AudioMothCode"] not in good_AM:
        intial_date = datetime.strptime(df.loc[i, "StartDateTime"], '%d.%m.%Y %H:%M')
        print(intial_date)
        print(i)
        add = add + 40
        if(add >= 100):
            break



