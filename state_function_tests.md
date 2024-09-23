# Tests:
## 1. I start with british, north, moderate
SYS: suggests one seven

Me: "another"

SYS: "I am not sure about the results. Could you provide anotherpreference?"

Me: "another"

SYS: "There is no restaurant with given parameters."

**After first another, the system should not ask any other prreferences according to the new state diagram. State5 goes to State4 when speechAct:reqalts with no arguments.**

**When there is only one restaurant, the system does the right job, states that there are no such restaurants. But ideally it should say that there are no OTHER restaurants.**

## 2. I start with british, north, moderate
SYS: suggests one seven

Me: "What about indian?"

SYS: speechAct: reqalts. But I can see both indian and british in preferences[food]. Suggests graffiti.

Me: "another"

SYS: "one seven"

Me: "another"

SYS: "tandoori"

Me: "another"

SYS: "one seven"

Me: "another"

SYS: "tandoori"

**It looks like when there are more than 1 restaurants, the system can continue suggesting the same restaurants forever. That is why I had created states 7, 8, 9.**

**I think once the user says "what about indian", the system should drop british from preferences[food] and put indian instead. This could work too but thats what I saw from the dialogs.**

## 3. I start with indian
Me: "british"

SYS: adds british to preferences[food], now theres indian, british

Me: "not british"

SYS: adds british to antipaties and deletes it from preferences[food]

Me: "british"

SYS: adds british back to preferences[food] but doesnt remove it from antipaties.

**I know I am the annoying tester here but I guess at the last step the system should remove british from antipaties, hehe.**

## 4. Lavenstien Distance
Me: "biritihs" edit_dist = 3

SYS: doesnt get british.

Me: "britihs" edit_dist = 2

SYS: doesnt get british

**The default value for max_distance=1, it should be 3 or 4 depending on the implementation. The document says if the distance is <= 3 then we should match the two words.**

## 5. I start with indian, north, moderate
SYS: "My suggestion is the nirala". 

Me: "any other"

SYS: speechAct: inform. "There is no restaurant with given parameters. Please provide any preferences again."

Me: "another"

SYS: speechAct: reqalts. "There is no restaurant with given parameters. Please provide any preferences again." But there is no change in preferences list.

Me: "another"

SYS: same

Me: "another"

SYS: same

Me: "chinese, moderate"

SYS: suggests graffiti. food: [indian, chinese]

**When I say "any other" if the system sees it as inform why does it reply no such restaurants?**

**When I try again with another, system understands that its reqalts but still replies no such restaurants. There are 4 other such restaurants which the system did not suggest yet.**

**After saying there are no such restaurants, when I say chinese, moderate, the system gives me an indian restaurant in accordance with my previous choices. This is why I believe we should keep only 1 value in preference fields.**

## 6. I start with indian, north, moderate
SYS: suggests tandoori place

Me: "this is indian, right?"

SYS: speechAct: confirm, "Tandoori place is an indianrestaurant"

Me: "it is expensive, correct?"

SYS: speechAct: confirm, but returns the address and postcode.

Me: "Whats the phone number?"

SYS: speechAct: request, returns phone number.

Me: Whats the address?

SYS: speechAct: request, return address AND ZIPCODE.

Me: whats the postcode?

SYS: speechAct
