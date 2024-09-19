# Some states:
* inform -> save information
* restaurant not found -> reqalts (only update given info)
* after not understanding reqalts twice, system made a random suggestion of italian instead of indian.
* after each inform -> check what if the info set is satisfied? If not, ask one of the renaining info's in random.
* affirm?
* Request(any_info) -> return which ever information is requested. Once a restaurant is selected, any field in the CSV can be asked.
* speech act = bye -> end dialog
* info ares can be "dontcare"
* speech act = null -> make another suggestion from the list (or keep asking for info) so generally we repeat the current state.


# General Info:
* We can get info through inform, confirm, reqalts


# Questions:
* Why does our system ask follow up questions. Like user says British restaurant, system asks: British, right? If it got the British why does it ask? Is there confidence? What kind of a state is that?
