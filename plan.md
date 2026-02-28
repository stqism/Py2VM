# Plan: Add Async & Generator Opcode Support to Py2VM

## Current State

Py2VM is a Python 3.11 bytecode interpreter with an explicit frame stack. It
currently supports Tiers 1–5 (arithmetic, containers, iteration, exceptions,
closures, classes, pattern matching, with-statements). **All generator and async
opcodes (Tier 6) are explicitly rejected** via the `_TIER6_OPCODES` frozenset
with a `NotImplementedError`.

The VM uses a `frames` list as a call stack, with `CALL` pushing new `Frame`
objects and `RETURN_VALUE` popping them. This stack-based model must be extended
to support **suspendable/resumable execution** for generators, coroutines, and
async generators.

## Target Opcodes (CPython 3.11)

| Opcode | Used By | Purpose |
|---|---|---|
| `RETURN_GENERATOR` | gen, coro, agen | Create generator/coroutine object from current frame, return it |
| `YIELD_VALUE` | gen, agen | Suspend frame, yield TOS to caller |
| `SEND` | yield-from, await | Send value into sub-iterator; jump on StopIteration |
| `GET_YIELD_FROM_ITER` | yield-from | Ensure TOS is an iterator (for `yield from`) |
| `GET_AWAITABLE` | await, async-with | Get `__await__` iterator from coroutine/awaitable |
| `ASYNC_GEN_WRAP` | agen | Wrap yielded value in `async_generator_asend` |
| `GET_AITER` | async-for | Call `__aiter__` on TOS |
| `GET_ANEXT` | async-for | Call `__anext__` on async iterator |
| `BEFORE_ASYNC_WITH` | async-with | Call `__aenter__`, push `__aexit__` |
| `END_ASYNC_FOR` | async-for | Handle `StopAsyncIteration` in async for loops |

## Code Object Flags

| Flag | Hex | Meaning |
|---|---|---|
| `CO_GENERATOR` | `0x20` | Function is a generator |
| `CO_COROUTINE` | `0x80` | Function is a native coroutine (`async def`) |
| `CO_ASYNC_GENERATOR` | `0x200` | Function is an async generator (`async def` + `yield`) |
| `CO_ITERABLE_COROUTINE` | `0x100` | Coroutine wrapped by `types.coroutine` |

---

## Implementation Plan

### Phase 1: Core Infrastructure — Generator Objects & Resumable Frames

**Goal:** Support simple `yield` and `next()` on generators.

#### Step 1.1: Add `VMGenerator` class

Create a generator object that wraps a suspended `Frame`:

```python
class VMGenerator:
    """Generator object — wraps a suspended Frame for resumable execution."""
    def __init__(self, frame, builtins):
        self._frame = frame       # The suspended Frame
        self._builtins = builtins
        self._started = False
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        if self._closed:
            raise StopIteration
        if not self._started:
            if value is not None:
                raise TypeError("can't send non-None value to a just-started generator")
            self._started = True
        # Resume the frame. The value gets pushed onto the frame's stack
        # (YIELD_VALUE pops TOS as the yielded value, then RESUME expects
        # the sent value to appear on the stack when execution continues).
        return _resume_generator(self, value)

    def throw(self, typ, val=None, tb=None):
        if self._closed:
            raise StopIteration
        if isinstance(typ, BaseException):
            exc = typ
        else:
            exc = typ(val) if val is not None else typ()
        return _throw_into_generator(self, exc)

    def close(self):
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError("generator ignored GeneratorExit")
        finally:
            self._closed = True
```

#### Step 1.2: Add `_resume_generator()` helper

This is the key function. It runs the generator's frame until it hits
`YIELD_VALUE` (suspend) or `RETURN_VALUE` (exhausted → `StopIteration`).

The main execution loop (`py2vm()`) currently operates on a `frames` list. The
generator resume function needs to run the same interpretation logic. Two
approaches:

**Approach chosen: Refactor the inner loop into a reusable `_run_frames()` function.**

Extract the `while frames:` loop body from `py2vm()` into a standalone
`_run_frames(frames, builtins, log)` function that returns a `(result_type,
value)` tuple:
- `('return', val)` — frame stack exhausted, final return value
- `('yield', val)` — hit YIELD_VALUE, frame is suspended

`py2vm()` calls `_run_frames()` for normal execution.
`_resume_generator()` wraps the generator's frame in a 1-element list and
calls `_run_frames()`.

#### Step 1.3: Implement `RETURN_GENERATOR` opcode

When the VM encounters `RETURN_GENERATOR` inside a function whose code object
has `CO_GENERATOR` set:
1. **Snapshot the current frame** (it's already set up with locals bound).
2. **Create a `VMGenerator`** wrapping that frame.
3. **Pop the frame** from the call stack.
4. **Push the `VMGenerator`** onto the caller's stack.

```
RETURN_GENERATOR:
    gen_frame = frames.pop()  # current frame, already initialized
    gen = VMGenerator(gen_frame, builtins)
    if frames:
        frames[-1].stack.append(gen)
    else:
        final_retval = gen
    continue
```

#### Step 1.4: Implement `YIELD_VALUE` opcode

When executing inside a generator frame:
1. Pop TOS as the yielded value.
2. Signal to `_run_frames()` to return `('yield', value)`.
3. The frame's `ip` is already advanced past YIELD_VALUE, so when resumed
   it will execute the next instruction (RESUME).

The `RESUME` opcode with arg=1 (after yield) is already a no-op, which is
correct.

#### Step 1.5: Handle `RETURN_VALUE` inside generators

When `RETURN_VALUE` fires in a generator frame (frame stack has 1 entry during
generator resume):
- The return value becomes the `StopIteration.value`.
- `_resume_generator()` raises `StopIteration(retval)`.
- Mark generator as closed.

#### Step 1.6: Handle the sent value

After `YIELD_VALUE` suspends, the next instruction is `RESUME 1` (no-op), then
whatever follows. The value sent via `.send()` should be **pushed onto the
frame's stack** before resuming (CPython pushes the sent value which RESUME
expects on TOS; then subsequent code like `STORE_FAST` picks it up).

Actually in CPython 3.11 bytecode: after `YIELD_VALUE`, the `RESUME 1` is a
no-op, then the *sent value* is what sits on TOS (it replaces the yield
expression's result). So `_resume_generator` must push `value` onto
`frame.stack` before resuming.

#### Step 1.7: Remove generator opcodes from `_TIER6_OPCODES`

Remove `RETURN_GENERATOR`, `YIELD_VALUE`, and `SEND` from the rejection set
as they are implemented.

---

### Phase 2: `yield from` and `SEND` / `GET_YIELD_FROM_ITER`

**Goal:** Support `yield from <iterable>` delegation.

#### Step 2.1: Implement `GET_YIELD_FROM_ITER`

```
GET_YIELD_FROM_ITER:
    TOS = top of stack
    if isinstance(TOS, VMGenerator) or isinstance(TOS, VMCoroutine):
        pass  # already an iterator
    else:
        TOS = iter(TOS)  # convert to iterator
```

#### Step 2.2: Implement `SEND` opcode

`SEND(target)` works as follows (CPython 3.11 semantics):
- TOS = value to send (the thing on top)
- TOS1 = the sub-iterator
- If TOS is None and sub-iterator supports `__next__`, call `__next__()`.
- Otherwise call sub-iterator's `.send(TOS)`.
- If the sub-call raises `StopIteration(val)`:
  - Pop both TOS and TOS1
  - Push `val` (the StopIteration's value)
  - Jump to `target`
- Otherwise (yielded a value):
  - Replace TOS with the yielded value (leave TOS1 = sub-iterator)
  - Fall through to next instruction (which is `YIELD_VALUE` to propagate yield up)

```python
SEND:
    value = stk.pop()      # sent value
    sub_iter = stk[-1]     # peek at sub-iterator
    try:
        if value is None:
            result = next(sub_iter)
        else:
            result = sub_iter.send(value)
        stk.append(result)  # will be yielded by next YIELD_VALUE
    except StopIteration as e:
        stk.pop()           # remove sub-iterator
        stk.append(e.value) # push StopIteration's value
        # Jump to target
        f.ip = f.offset_to_index[<target_offset>]
```

The target offset for SEND needs to be computed. In CPython 3.11, `SEND arg`
jumps forward by `arg` instructions. We need to compute the target byte offset
from the instruction's offset + (arg * 2) and look it up in `offset_to_index`.

Actually, looking at the `dis` output: `SEND 3 (to 24)` — the `argval` already
contains the target offset (24). So we use `f.offset_to_index[argval]` as the
jump target.

---

### Phase 3: Coroutines (`async def`)

**Goal:** Support `async def` coroutines, `await`, and running them.

#### Step 3.1: Add `VMCoroutine` class

Very similar to `VMGenerator` but:
- Implements `__await__()` which returns `self` (the coroutine is its own
  iterator for the await protocol).
- Has `send()`, `throw()`, `close()` like generators.
- Distinguished by checking `CO_COROUTINE` flag (0x80).

```python
class VMCoroutine:
    """Coroutine object — wraps a suspended Frame."""
    def __init__(self, frame, builtins):
        self._frame = frame
        self._builtins = builtins
        self._started = False
        self._closed = False

    def __await__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value): ...  # same as VMGenerator.send
    def throw(self, ...): ...   # same as VMGenerator.throw
    def close(self): ...        # same as VMGenerator.close
```

#### Step 3.2: Update `RETURN_GENERATOR` for coroutines

Check code flags:
- `CO_COROUTINE` (0x80) → create `VMCoroutine`
- `CO_GENERATOR` (0x20) → create `VMGenerator`
- `CO_ASYNC_GENERATOR` (0x200) → create `VMAsyncGenerator` (Phase 4)

#### Step 3.3: Implement `GET_AWAITABLE` opcode

```
GET_AWAITABLE(where):
    TOS = top of stack
    if isinstance(TOS, VMCoroutine):
        result = TOS  # coroutines are their own await iterators
    elif hasattr(TOS, '__await__'):
        result = TOS.__await__()
    else:
        raise TypeError("object ... can't be used in 'await' expression")
    Replace TOS with result
```

The `where` argument (0, 1, or 2) indicates context (await expr, async with
enter, async with exit) but doesn't affect runtime behavior for our purposes.

#### Step 3.4: Provide a simple event loop / `run()` function

For users to actually execute coroutines, provide a minimal `vm_run()` helper:

```python
def vm_run(coro):
    """Minimal event loop: drive a coroutine to completion."""
    result = None
    try:
        while True:
            result = coro.send(result)
    except StopIteration as e:
        return e.value
```

This is analogous to `asyncio.run()` but without actual I/O scheduling. It's
sufficient for non-I/O coroutines and testing. For the test suite, we can also
make coroutines callable in the VM test harness by wrapping calls with this
runner.

---

### Phase 4: Async Generators

**Goal:** Support `async def` functions that contain `yield`.

#### Step 4.1: Add `VMAsyncGenerator` class

```python
class VMAsyncGenerator:
    def __init__(self, frame, builtins):
        self._frame = frame
        self._builtins = builtins
        self._started = False
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Returns an awaitable that yields the next value
        return self.asend(None)

    def asend(self, value):
        return VMAsyncGenASend(self, value)

    def athrow(self, typ, val=None, tb=None): ...
    def aclose(self): ...
```

The `VMAsyncGenASend` is an awaitable object whose `__await__` drives the
generator frame forward.

#### Step 4.2: Implement `ASYNC_GEN_WRAP` opcode

In async generators, yielded values get wrapped so the async-for protocol can
distinguish yields from returns:

```
ASYNC_GEN_WRAP:
    TOS = wrap TOS in an async_generator_wrapped_value marker
```

We can use a simple wrapper class:

```python
class _AsyncGenWrappedValue:
    __slots__ = ('value',)
    def __init__(self, value):
        self.value = value
```

When `YIELD_VALUE` is hit after `ASYNC_GEN_WRAP`, the yielded value is this
wrapper. The `VMAsyncGenASend.__next__` method unwraps it.

---

### Phase 5: Async For / Async With Opcodes

**Goal:** Support `async for` and `async with` inside coroutines.

#### Step 5.1: Implement `GET_AITER`

```
GET_AITER:
    TOS = TOS.__aiter__()
```

#### Step 5.2: Implement `GET_ANEXT`

```
GET_ANEXT:
    Push TOS.__anext__() onto stack (don't pop the async iterator)
```

#### Step 5.3: Implement `BEFORE_ASYNC_WITH`

```
BEFORE_ASYNC_WITH:
    mgr = TOS (pop)
    exit_fn = mgr.__aexit__   # push this first (for later cleanup)
    enter_result = mgr.__aenter__()  # push this (will be awaited)
    push exit_fn, then enter_result
```

#### Step 5.4: Implement `END_ASYNC_FOR`

```
END_ASYNC_FOR:
    TOS = the exception
    TOS1 = the async iterator
    if isinstance(TOS, StopAsyncIteration):
        pop both, continue (exit the for loop)
    else:
        re-raise TOS
```

---

### Phase 6: `SETUP_ANNOTATIONS` Opcode

This is unrelated to generators/async but is in the Tier 6 rejection set.

```
SETUP_ANNOTATIONS:
    if '__annotations__' not in f.name_dict:
        f.name_dict['__annotations__'] = {}
    # Also handle: if in locals, set there too
```

---

### Phase 7: Tests

#### Step 7.1: Generator tests

```python
# Basic generator iteration
def gen():
    yield 1
    yield 2
    yield 3
print(list(gen()))  # [1, 2, 3]

# Generator with send
def echo():
    val = yield 'ready'
    while val is not None:
        val = yield f'echo: {val}'
g = echo()
print(next(g))        # 'ready'
print(g.send('hello'))# 'echo: hello'
print(g.send('world'))# 'echo: world'

# Generator with return value
def gen_ret():
    yield 1
    return 42
g = gen_ret()
next(g)
try:
    next(g)
except StopIteration as e:
    print(e.value)  # 42

# yield from delegation
def inner():
    yield 1
    yield 2
    return 'done'
def outer():
    result = yield from inner()
    print(result)
    yield 3
print(list(outer()))  # prints 'done', then [1, 2, 3]

# Generator expressions
print(sum(x*x for x in range(5)))  # 30

# Generator close/throw
def gen_close():
    try:
        yield 1
        yield 2
    except GeneratorExit:
        pass  # normal cleanup
g = gen_close()
next(g)
g.close()
print('closed ok')

# Nested yield from with send
def bottom():
    val = yield 'bottom'
    return val * 2
def middle():
    result = yield from bottom()
    return result + 10
def top():
    result = yield from middle()
    yield result
g = top()
print(next(g))        # 'bottom'
print(g.send(5))      # 20 (5*2+10)
```

#### Step 7.2: Coroutine tests

```python
# Basic coroutine
async def simple():
    return 42
# Need to drive with our simple event loop or manually

# Await another coroutine
async def add(a, b):
    return a + b
async def main():
    result = await add(3, 4)
    print(result)

# async with
# async for
```

#### Step 7.3: Async generator tests

```python
async def async_count(n):
    for i in range(n):
        yield i

async def main():
    result = []
    async for x in async_count(5):
        result.append(x)
    print(result)
```

---

## File Changes Summary

| File | Changes |
|---|---|
| `py2vm.py` | Add `VMGenerator`, `VMCoroutine`, `VMAsyncGenerator` classes; extract `_run_frames()`; implement all 10 Tier 6 opcodes; update `RETURN_GENERATOR` to detect type from flags; remove opcodes from `_TIER6_OPCODES` as implemented; add `vm_run()` helper |
| `test_vm.py` | Add comprehensive test cases for generators, yield-from, coroutines, async generators, async-for, async-with |

## Implementation Order (prioritized by dependency)

1. **Refactor**: Extract main loop into `_run_frames()` (prerequisite for everything)
2. **VMGenerator** + `RETURN_GENERATOR` + `YIELD_VALUE` + `RETURN_VALUE` in generator context
3. **Generator `.send()`, `.throw()`, `.close()`**
4. **`GET_YIELD_FROM_ITER`** + **`SEND`** (for `yield from`)
5. **VMCoroutine** + `GET_AWAITABLE` + `vm_run()` helper
6. **VMAsyncGenerator** + `ASYNC_GEN_WRAP` + async gen helpers
7. **`GET_AITER`** + **`GET_ANEXT`** + **`BEFORE_ASYNC_WITH`** + **`END_ASYNC_FOR`**
8. **`SETUP_ANNOTATIONS`**
9. **Tests** for each phase
10. **Update header comment** to reflect Tier 6 support

## Risks & Considerations

- **Frame snapshotting**: Generator frames need to preserve their entire state
  (stack, ip, locals, cells, block_stack) across suspensions. Our `Frame` class
  already stores all of this, so suspension is essentially just "stop running and
  keep the frame object alive."
- **Exception propagation through yield-from chains**: `SEND`/`throw()` must
  correctly propagate exceptions through nested yield-from delegation. This is
  the most complex part semantically.
- **Interaction with exception tables**: Generator frames that receive
  `.throw()` need the exception table handling to work correctly when the
  exception is injected.
- **`__del__` / finalization**: CPython generators have `__del__` that calls
  `.close()`. Our VM generators won't have automatic finalization — this is
  acceptable as a known limitation.
- **`asyncio` compatibility**: Full asyncio support is out of scope. The
  `vm_run()` helper provides enough to test coroutine mechanics without a real
  event loop.
