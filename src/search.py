from SPARQLWrapper import SPARQLWrapper, CSV


class Search:

    def __init__(self):
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.sparql.addDefaultGraph("http://dbpedia.org")
        self.new = []
        self.existing = []
        # this list is for test, Should be replaced with the list of the words we found in the previous step!
        # mylist = [("London", "Location"), ("Obama", "Person"), ("UNICEF", "Organization"), ("jhfvjhegrfvh", "Location"
        # ), ("Noor_Nakhaei", "Person"), ("Center_For_Smart_Health", "Organization")]

    def search(self, mylist):
        # we search in the dbpedia to see if the words exist
        for i in mylist:
            self.sparql.setQuery("""
            SELECT DISTINCT ?item ?label WHERE{
                        ?item rdfs:label ?label .
                        FILTER (lang(?label) = 'en').
                        ?label bif:contains '%s' .
                        ?item dct:subject ?sub
                }
            """ % i[0])
            try:
                self.sparql.setReturnFormat(CSV)
                results = self.sparql.query()
                triples = results.convert()
            except:
                print("query failed")
            # if the word exists, we add it to the to existing list
            if len(triples) > 15:
                print(triples)
                self.existing.append(i)
            # if the word doesn't exists, we add it to the to new list
            else:
                self.new.append(i)
        return self.new, self.existing
