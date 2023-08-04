# Custom-chatbot----Open-source-Vector-DB-Embeddings

Here is the overall architecure of Vector DB and how it works!






Vector DB Architecture:

![image](https://github.com/puneethegde/Custom-chatbot----Open-source-Vector-DB-Embeddings/assets/88820961/8a7bc642-2bdb-4631-9721-d07613218e3f)






Unstructured data is fed into langchain library which will divide the whole document into smaller chunks based on value we have given.
1.	Divide into smaller chunks:      
•	Split by character.
•	Recursive character Splitter (Efficient one)
•	Split by tokens.
•	Split by character: This method is the simplest to implement, but it is also the least efficient. It requires looping through the entire string and checking each character. This can be slow, especially for long strings.
•	Recursive character splitter: This method is more efficient than split by character, but it is still not as efficient as split by tokens. It uses recursion to split the string into smaller and smaller pieces until it is split into individual characters. This can be faster than looping through the entire string, but it can still be slow for long strings.
•	Split by tokens: This method is the most efficient way to split a string. It uses a regular expression to split the string into tokens. This can be very fast, even for long strings.
Splitting method	Efficiency	Complexity	Use cases
Split by character	Least efficient	Simple	Simple strings
Recursive character splitter	More efficient	Medium	Complex strings
Split by tokens	Most efficient	Complex	Strings with regular expressions
Here are some additional considerations when choosing a splitting method:
•	The length of the string: If the string is very long, then split by tokens will be the most efficient method.
•	The complexity of the string: If the string contains regular expressions, then split by tokens will be the most efficient method.
•	The specific use case: If there is a need to split the string into individual characters, then split by character is the best option.
 
 


