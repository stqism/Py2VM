"""Differential testing harness: run code on CPython and py2vm, compare results."""
import sys
import traceback
import io


def run_cpython(source):
    """Execute source on CPython, capture stdout and return (stdout, globals, exc)."""
    ns = {}
    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    exc = None
    try:
        exec(compile(source, '<test>', 'exec'), ns)
    except Exception as e:
        exc = e
    finally:
        sys.stdout = old_stdout
    return buf.getvalue(), ns, exc


def run_vm(source):
    """Execute source through py2vm, capture stdout and return (stdout, exc)."""
    from py2vm import buildcode
    exc = None
    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        buildcode(source)
    except Exception as e:
        exc = e
    finally:
        sys.stdout = old_stdout
    return buf.getvalue(), exc


def diff_test(label, source, expect_exception=False):
    """Run source on both engines, compare. Returns True if matching."""
    cpython_stdout, cpython_ns, cpython_exc = run_cpython(source)
    vm_stdout, vm_exc = run_vm(source)

    passed = True
    reasons = []

    if expect_exception:
        if cpython_exc is not None and vm_exc is None:
            reasons.append("CPython raised %s but VM did not" % type(cpython_exc).__name__)
            passed = False
        elif cpython_exc is None and vm_exc is None:
            pass  # neither raised, fine
        elif cpython_exc is not None and vm_exc is not None:
            if type(cpython_exc).__name__ != type(vm_exc).__name__:
                reasons.append("Exception type mismatch: CPython=%s VM=%s" % (
                    type(cpython_exc).__name__, type(vm_exc).__name__))
                passed = False
    else:
        if cpython_exc is not None:
            reasons.append("CPython raised unexpectedly: %s" % cpython_exc)
            passed = False
        if vm_exc is not None:
            reasons.append("VM raised unexpectedly: %s" % vm_exc)
            passed = False

    # Compare captured stdout
    if passed:
        if vm_stdout != cpython_stdout:
            reasons.append("Output mismatch:\n  CPython: %r\n  VM:      %r" % (
                cpython_stdout, vm_stdout))
            passed = False

    if passed:
        print("PASS [%s]" % label)
    else:
        print("FAIL [%s]" % label)
        for r in reasons:
            print("  -> %s" % r)

    return passed


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
TESTS = [
    # Tier 1: basic operations
    ("arithmetic", "x = 2 + 3 * 4 - 1\nprint(x)\n"),
    ("unary_ops", "x = -5\nprint(+x, -x, not x, ~x)\n"),
    ("binary_op_all", """
print(3 + 2)
print(3 - 2)
print(3 * 2)
print(7 // 2)
print(7 % 2)
print(2 ** 3)
print(7 / 2)
print(3 & 1)
print(3 | 4)
print(3 ^ 1)
print(1 << 3)
print(16 >> 2)
"""),
    ("comparison", """
print(1 < 2, 2 <= 2, 3 == 3, 4 != 5)
print(5 > 4, 5 >= 5, 1 is 1, 1 is not 2)
"""),
    ("is_op", "print(None is None, None is not None)\n"),
    ("contains_op", "print(1 in [1,2,3], 4 not in [1,2,3])\n"),
    ("simple_function", """
def f(a, b):
    return a + b
print(f(3, 4))
"""),
    ("function_defaults", """
def f(a, b=10):
    return a + b
print(f(1))
print(f(1, 2))
"""),
    ("nested_calls", """
def double(x):
    return x * 2
def add(a, b):
    return a + b
print(add(double(3), double(4)))
"""),
    ("global_var", """
x = 10
def f():
    global x
    x = 20
f()
print(x)
"""),

    # Tier 2: containers and iteration
    ("for_loop", """
s = 0
for i in range(10):
    s += i
print(s)
"""),
    ("build_list", "print([1, 2, 3])\n"),
    ("build_tuple", "print((1, 2, 3))\n"),
    ("build_set", "print(sorted({3, 1, 2}))\n"),
    ("build_dict", "print({'a': 1, 'b': 2})\n"),
    ("subscr", """
x = [10, 20, 30]
print(x[1])
x[1] = 99
print(x)
"""),
    ("unpack", """
a, b, c = 1, 2, 3
print(a, b, c)
"""),
    ("list_comp", "print([x*2 for x in range(5)])\n"),
    ("dict_comp", "print({k: k*2 for k in range(3)})\n"),
    ("set_comp", "print(sorted({x % 3 for x in range(10)}))\n"),
    ("nested_list", """
xs = [1, 2, 3]
ys = [*xs, 4, 5]
print(ys)
"""),
    ("nested_dict", """
a = {'x': 1}
b = {**a, 'y': 2}
print(sorted(b.items()))
"""),
    ("for_else", """
for i in range(3):
    pass
else:
    print('done')
"""),

    # Tier 3: exceptions
    ("try_except", """
try:
    1/0
except ZeroDivisionError:
    print('caught')
"""),
    ("try_except_else", """
try:
    x = 1
except:
    print('bad')
else:
    print('ok')
"""),
    ("raise_basic", """
try:
    raise ValueError('oops')
except ValueError as e:
    print(str(e))
"""),

    # Tier 4: closures
    ("closure_basic", """
def outer(x):
    def inner():
        return x
    return inner
print(outer(42)())
"""),
    ("closure_counter", """
def make_counter():
    count = 0
    def inc():
        nonlocal count
        count += 1
        return count
    return inc
c = make_counter()
print(c(), c(), c())
"""),

    # Tier 4: classes
    ("class_basic", """
class MyClass:
    val = 10
    def get_val(self):
        return self.val
obj = MyClass()
print(obj.get_val())
"""),

    # Tier 5: quality of life
    ("fstring", """
name = 'world'
print(f'hello {name}')
"""),
    ("slice_ops", """
x = [0, 1, 2, 3, 4]
print(x[1:3])
print(x[::2])
"""),
    ("star_args", """
def f(*args, **kwargs):
    return (args, sorted(kwargs.items()))
print(f(1, 2, a=3))
"""),
    ("kwonly_args", """
def f(a, *, b=10):
    return a + b
print(f(1, b=20))
print(f(5))
"""),
    ("inplace_ops", """
x = 10
x += 5
x -= 3
x *= 2
print(x)
"""),
    ("bool_shortcircuit", """
def side(x):
    print(x, end=' ')
    return x
print(side(0) or side(1))
print(side(1) and side(2))
"""),
    ("while_loop", """
i = 0
s = 0
while i < 5:
    s += i
    i += 1
print(s)
"""),
    ("if_elif_else", """
x = 15
if x > 20:
    print('big')
elif x > 10:
    print('medium')
else:
    print('small')
"""),
    ("multiple_return", """
def classify(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    return 'zero'
print(classify(5), classify(-3), classify(0))
"""),
    ("string_methods", """
s = 'hello world'
print(s.upper())
print(s.split())
print(len(s))
"""),
    ("import_basic", """
import math
print(math.gcd(12, 8))
"""),
    ("delete_ops", """
x = [1, 2, 3]
del x[1]
print(x)
"""),
    # ------------------------------------------------------------------
    # Tier 1: with-statements and exception handling
    # ------------------------------------------------------------------
    ("with_statement", """
class CM:
    def __enter__(self):
        print('enter')
        return self
    def __exit__(self, *args):
        print('exit')
        return False
with CM() as c:
    print('body')
"""),
    ("with_exception", """
class CM:
    def __enter__(self):
        return self
    def __exit__(self, typ, val, tb):
        print(f'exit: {typ.__name__}')
        return True
with CM():
    raise ValueError('oops')
print('after')
"""),
    ("with_reraise", """
class NonSuppressor:
    def __enter__(self): return self
    def __exit__(self, *a): return False
try:
    with NonSuppressor():
        raise ValueError('not suppressed')
except ValueError as e:
    print(str(e))
"""),
    ("try_finally", """
def f():
    try:
        print('try')
        return 1
    finally:
        print('finally')
r = f()
print(r)
"""),
    ("nested_with", """
class CM:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print(f'enter {self.name}')
        return self
    def __exit__(self, *a):
        print(f'exit {self.name}')
        return False
with CM('a') as a, CM('b') as b:
    print('body')
"""),
    # ------------------------------------------------------------------
    # Tier 2: closures / cells
    # ------------------------------------------------------------------
    ("class_with_closure", """
def make():
    x = 10
    class C:
        val = x
    return C
print(make().val)
"""),
    # ------------------------------------------------------------------
    # Tier 3: extended unpacking
    # ------------------------------------------------------------------
    ("unpack_ex", """
a, *b, c = [1, 2, 3, 4, 5]
print(a, b, c)
first, *rest = 'hello'
print(first, rest)
*init, last = range(4)
print(init, last)
"""),
    ("unpack_ex_empty_star", """
a, *b, c = [1, 2]
print(a, b, c)
"""),
    # ------------------------------------------------------------------
    # Tier 4: assert
    # ------------------------------------------------------------------
    ("assert_pass", """
x = 42
assert x > 0
print('ok')
"""),
    ("assert_fail", """
try:
    assert False, 'nope'
except AssertionError as e:
    print(f'caught: {e}')
"""),
    # ------------------------------------------------------------------
    # Tier 5: pattern matching
    # ------------------------------------------------------------------
    ("match_literal", """
def describe(x):
    match x:
        case 1:
            return 'one'
        case 2:
            return 'two'
        case _:
            return 'other'
print(describe(1))
print(describe(2))
print(describe(99))
"""),
    ("match_sequence", """
def f(x):
    match x:
        case [a, b]:
            return f'pair: {a},{b}'
        case [a, b, c]:
            return f'triple: {a},{b},{c}'
        case _:
            return 'other'
print(f([1, 2]))
print(f([1, 2, 3]))
print(f('hi'))
"""),
    ("match_mapping", """
def f(x):
    match x:
        case {'action': 'go', 'dir': d}:
            return f'go {d}'
        case {'action': 'stop'}:
            return 'stop'
        case _:
            return 'unknown'
print(f({'action': 'go', 'dir': 'north'}))
print(f({'action': 'stop'}))
print(f(42))
"""),
    ("match_class", """
class Point:
    __match_args__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y
def f(p):
    match p:
        case Point(x, y) if x > 0:
            return f'pos: {x},{y}'
        case Point(x, y):
            return f'other: {x},{y}'
print(f(Point(1, 2)))
print(f(Point(-1, 3)))
"""),

    # ------------------------------------------------------------------
    # Tier 6: Generators
    # ------------------------------------------------------------------
    ("gen_basic", """
def gen():
    yield 1
    yield 2
    yield 3
print(list(gen()))
"""),
    ("gen_send", """
def echo():
    val = yield 'ready'
    while val is not None:
        val = yield f'echo: {val}'
g = echo()
print(next(g))
print(g.send('hello'))
print(g.send('world'))
"""),
    ("gen_return_value", """
def gen():
    yield 1
    return 42
g = gen()
print(next(g))
try:
    next(g)
except StopIteration as e:
    print(e.value)
"""),
    ("gen_close", """
def gen():
    try:
        yield 1
        yield 2
    except GeneratorExit:
        pass
g = gen()
print(next(g))
g.close()
print('closed ok')
"""),
    ("gen_throw", """
def gen():
    try:
        yield 1
    except ValueError as e:
        yield f'caught: {e}'
g = gen()
print(next(g))
print(g.throw(ValueError, 'oops'))
"""),
    ("gen_expression", """
print(sum(x*x for x in range(5)))
print(list(x for x in range(4) if x % 2 == 0))
"""),
    ("gen_for_iteration", """
def count_up(n):
    i = 0
    while i < n:
        yield i
        i += 1
result = []
for x in count_up(5):
    result.append(x)
print(result)
"""),
    ("gen_multiple_yields", """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
print(list(fib(8)))
"""),
    ("gen_nested", """
def inner():
    yield 'a'
    yield 'b'
def outer():
    for x in inner():
        yield x.upper()
    yield 'C'
print(list(outer()))
"""),

    # ------------------------------------------------------------------
    # Tier 6: yield from
    # ------------------------------------------------------------------
    ("yield_from_basic", """
def inner():
    yield 1
    yield 2
    return 'done'
def outer():
    result = yield from inner()
    print(result)
    yield 3
print(list(outer()))
"""),
    ("yield_from_list", """
def gen():
    yield from [10, 20, 30]
print(list(gen()))
"""),
    ("yield_from_range", """
def gen():
    yield from range(5)
print(list(gen()))
"""),
    ("yield_from_send", """
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
print(next(g))
print(g.send(5))
"""),
    ("yield_from_chain", """
def gen1():
    yield 1
    yield 2
def gen2():
    yield from gen1()
    yield 3
def gen3():
    yield from gen2()
    yield 4
print(list(gen3()))
"""),

    # ------------------------------------------------------------------
    # Tier 6: Coroutines (async/await)
    # ------------------------------------------------------------------
    ("coroutine_basic", """
from py2vm import vm_run
async def add(a, b):
    return a + b
async def main():
    result = await add(3, 4)
    return result
print(vm_run(main()))
"""),
    ("coroutine_nested_await", """
from py2vm import vm_run
async def double(x):
    return x * 2
async def add_doubled(a, b):
    da = await double(a)
    db = await double(b)
    return da + db
async def main():
    print(await add_doubled(3, 4))
vm_run(main())
"""),
    ("coroutine_try_except", """
from py2vm import vm_run
async def failing():
    raise ValueError('async error')
async def main():
    try:
        await failing()
    except ValueError as e:
        print(f'caught: {e}')
vm_run(main())
"""),

    # ------------------------------------------------------------------
    # Tier 6: Async generators
    # ------------------------------------------------------------------
    ("async_gen_basic", """
from py2vm import vm_run
async def async_count(n):
    for i in range(n):
        yield i
async def main():
    result = []
    async for x in async_count(5):
        result.append(x)
    print(result)
vm_run(main())
"""),
    ("async_gen_with_await", """
from py2vm import vm_run
async def compute(x):
    return x * 10
async def async_mapped(n):
    for i in range(n):
        val = await compute(i)
        yield val
async def main():
    result = []
    async for x in async_mapped(4):
        result.append(x)
    print(result)
vm_run(main())
"""),

    # ------------------------------------------------------------------
    # Tier 6: async with
    # ------------------------------------------------------------------
    ("async_with_basic", """
from py2vm import vm_run
class AsyncCM:
    async def __aenter__(self):
        print('enter')
        return self
    async def __aexit__(self, *args):
        print('exit')
        return False
async def main():
    async with AsyncCM() as cm:
        print('body')
vm_run(main())
"""),
    ("async_with_exception", """
from py2vm import vm_run
class AsyncCM:
    async def __aenter__(self):
        return self
    async def __aexit__(self, typ, val, tb):
        print(f'exit: {typ.__name__}')
        return True
async def main():
    async with AsyncCM():
        raise ValueError('oops')
    print('after')
vm_run(main())
"""),
]


def main():
    passed = 0
    failed = 0
    errors = []
    for label, source in TESTS:
        try:
            expect_exc = 'raise' in label and 'except' not in source
            if diff_test(label, source, expect_exception=expect_exc):
                passed += 1
            else:
                failed += 1
                errors.append(label)
        except Exception as e:
            print("ERROR [%s]: %s" % (label, e))
            traceback.print_exc()
            failed += 1
            errors.append(label)

    print("\n%d/%d tests passed" % (passed, passed + failed))
    if errors:
        print("Failed: %s" % ', '.join(errors))
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
