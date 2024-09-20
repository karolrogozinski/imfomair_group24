# Some states:
* inform -> save information
* restaurant not found -> reqalts (only update given info)
* after not understanding reqalts twice, system made a random suggestion of italian instead of indian.
* after each inform -> check what if the info set is satisfied? If not, ask one of the renaining info's in random.
* affirm?
* Request(any_info) -> return which ever information is requested. Once a restaurant is selected, any field in the CSV can be asked. If multiple fields are asked, conjunct the with and. Also create in generic but proper sentences. Eg: The phone number of \<restaurant\> is \<phone number\>.
* speech act = thankyou | bye -> end dialog
* info ares can be "dontcare"
* speech act = null -> make another suggestion from the list (or keep asking for info) so generally we repeat the current state. But I've seen skipping info and starting to suggest without price info too: Task: 12396
* reqalts(no new info) -> make a suggestion with the same preferences.
* reqalts(new info) -> change new info fields, make a new suggestion.
* I guess the user asks for alternative only after receiving a suggestion. So reqalts is useful only after suggestion state. But it can come anytime as the user can always be typing randomly.
* speech act = affirm -> proceed current state. I am not sure we would receive any affirms. Since in the dialogues affirm usually comes after the system asks a question.



# General Info:
* We can get info through inform, confirm, reqalts, negate


# Questions:
* Why does our system ask follow up questions. Like user says British restaurant, system asks: British, right? If it got the British why does it ask? Is there confidence? What kind of a state is that?
* There's a weird type attribute the system always fills with "restaurant". I think it's useless.
* What about task=find? I think that's useless too.


# Look Agains:
* How does the system behave after speechAct: null?
* negate (06417)
* confirm (06417, 05628) does it change the suggestion after a confimation?
* It looks like if there's an inform after suggestion, the system does not change suggestion but repeat relevant information to the user: (06417, 12396)
* Does speechAct thankyou end dialog alone (02286)
* affirm (02826)
* sometimes the system makes a suggestion without ever asking for a missing info field. (01932) Maybe we can do: if at least 1 info field is known then make a suggestion with x% prob and ask for missing fields with 100-x% probability.
* 
