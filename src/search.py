from SPARQLWrapper import SPARQLWrapper, CSV
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.addDefaultGraph("http://dbpedia.org")
# this list is for test, Should be replaced with the list of the words we found in the previous step!
mylist = [("London", "Location"), ("Obama", "Person"), ("UNICEF", "Organization"), ("jhfvjhegrfvh", "Location"), ("Noor_Nakhaei", "Person"), ("Center_For_Smart_Health", "Organization")]
new = []
existing = []
# we search in the dbpedia to see if the words exist
for i in mylist:
    sparql.setQuery("""
    SELECT DISTINCT ?item ?label WHERE{
                ?item rdfs:label ?label .
                FILTER (lang(?label) = 'en').
                ?label bif:contains '%s' .
                ?item dct:subject ?sub
        }
    """ % i[0])
    try:
        sparql.setReturnFormat(CSV)
        results = sparql.query()
        triples = results.convert()
    except:
        print("query failed")
    # if the word exists, we add it to the to existing list
    if len(triples) > 15:
        print(triples)
        existing.append(i)
    # if the word doesn't exists, we add it to the to new list
    else:
        new.append(i)

print(new)
print(existing)
