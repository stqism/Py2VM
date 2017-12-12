# Py2VM
Some semi-recent py2vm shit before getting bored


Source of opcodes:

https://docs.python.org/2/library/dis.html

How to add an opcode:

```
elif opcode_value == 'JUMP_ABSOLUTE':
            i = arg - 1
            if __INTERNAL__DEBUG_LOG:
```        
           
Add a line like that, `const_stack.insert(0, "test")` adds test to Top Of Stack.

`name_dict[]` is a list of strings already loaded to its own stack.

`arg` is the line.

`bytecode.co_consts` is a const stack. Note: what to do is documented at the opcode documentation 
