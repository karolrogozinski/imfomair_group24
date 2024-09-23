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
