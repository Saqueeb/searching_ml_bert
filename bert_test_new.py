#Importing the bert file
from bert import*
# taking the queries
query = ['Apple iPhone 8 pro max, 128GB, Gold - Fully Unlocked (Renewed)', 'La casa De Papel']  #query order = tittle(!)->brand->description
#taking the data array
df = [
    [u'Apple iPhone 8, 64GB, Gold - Fully Unlocked (Renewed)', u' Amazon Renewed', '"Iphone 8 introduces an all-new glass design. The world\'s most popular camera, now even better. The smartest, most powerful chip ever in a smartphone. Wireless charging that\'s truly effortless. And augmented reality experiences never before possible. Iphone 8. A new generation of iPhone.\\n\\n"', 'Review: this iphone is good'],
    [u'Apple iPhone X, 64GB, Space Gray - Fully Unlocked (Renewed)', u' Amazon Renewed', '"Apple iPhone X, 64GB, Space Gray - Fully Unlocked (Renewed)\\n\\n"'],
    [u'Apple iPhone 8 Plus, 64GB, Gold - Fully Unlocked (Renewed)', u' Amazon Renewed', '"This is a Fully Unlocked device, works with all carriers.\\n\\n"']
]   #Data array            remember!!!! there should data for tittle, brand and description, if you don't find any, put an empty string!


sub_category = ['iphone', '8', '64gb'] #subcategory will be compared with the best match found in data array
                                        #so be careful! put only important keywords that should be double checked with the result!

print(bert_test_main(query, df, sub_category))

