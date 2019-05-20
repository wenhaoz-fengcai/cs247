from SPARQLWrapper import SPARQLWrapper, CSV


class Search:
    
    """
    Python class which searches for the entities found in previous step, in the dbpedia database.
        If the entities were not found, they're declared in a list as new entities.
    Attributes:
        entities: a list of tuples: words, and they're type.
        new: a list of new entities.
        existing: a list of existing entities.
        sparql: This is a wrapper around a SPARQL service. It helps in creating the query URI and, possibly,
            convert the result into a more manageable format.
    """
    def __init__(self):
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.sparql.addDefaultGraph("http://dbpedia.org")
        self.new = []
        self.existing = []
        # this list is for test, Should be replaced with the list of the words we found in the previous step!
        # mylist = [("London", "Location"), ("Obama", "Person"), ("UNICEF", "Organization"), ("jhfvjhegrfvh", "Location"
        # ), ("Noor_Nakhaei", "Person"), ("Center_For_Smart_Health", "Organization")]

    def search(self, entities):
        # we search in the dbpedia to see if the words exist
        for i in entities:
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
