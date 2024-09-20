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
* reqalts(no new info) -> make a suggestion with the same preferences, but a different restaurant. (07527 always gives same restaurant, maybe we dont have to make it a different restaurant.)
* reqalts(new info) -> change new info fields, make a new suggestion, but a different restaurant.
* I guess the user asks for alternative only after receiving a suggestion. So reqalts is useful only after suggestion state. But it can come anytime as the user can always be typing randomly.
* speech act = affirm -> proceed current state. I am not sure we would receive any affirms. Since in the dialogues affirm usually comes after the system asks a question.
* speech act = restart -> repeat the last response. Thats what is done in the dialog data.



# General Info:
* We can get info through inform, confirm, reqalts, negate


# Questions:
* Why does our system ask follow up questions. Like user says British restaurant, system asks: British, right? If it got the British why does it ask? Is there confidence? What kind of a state is that?
* There's a weird type attribute the system always fills with "restaurant". I think it's useless.
* What about task=find? I think that's useless too.
* Task 06492: the system cant fully understand "care" prompt and asks follow up questions. After the follow-up questions it still knows what is the subject at that point. It's an important example but I think it can be the work of the next week.


# Look Agains:
* How does the system behave after speechAct: null? Sometimes reulsts in a general statement about the restaurant that was just suggested.
* negate (06417, 06492)
* confirm (06417, 05628) does it change the suggestion after a confimation?
* It looks like if there's an inform after suggestion, the system does not change suggestion but repeat relevant information to the user: (06417, 12396)
* weird example: 05697
* sometimes dontcare comes after a suggestion, such dontcares are not about any info field (08892), separate dontcare. In this case system makes another random suggestion.
* Does speechAct thankyou end dialog alone (02286)
* affirm (02826)
* sometimes the system makes a suggestion without ever asking for a missing info field. (01932) Maybe we can do: if at least 1 info field is known then make a suggestion with x% prob and ask for missing fields with 100-x% probability.
* with reqalts
* Sometimes when an info field changes, the system asks "Sorry would you like korean or belgian food?" (11541)
* If the system makes wrong deduction twice, and realizes that asks if it is right which results in an affirm or a negate. (08526)
* After reqalts, the system should refer to a different restaurant. If there are no other restaurants for a certain set of information, it w-should return no more restaurants. (08526)
* it looks like if the system receives two nulls in a row, it asks a question (00792 but its a very weird dialog there)
* 
