# optimizer.py — Bytecode optimizer for Py2VM
# Produces an optimized IR that the VM executes instead of raw bytecode.
# Never mutates code.co_code; instead generates a new instruction stream.

import dis as _dis_mod
import opcode as _opcode_mod
import weakref as _weakref_mod
import sys as _sys_mod

# ---------------------------------------------------------------------------
# Phase 1: Canonical IR
# ---------------------------------------------------------------------------

# Instruction flags (bitfield)
F_JUMP = 0x01        # Has a jump target
F_MAY_RAISE = 0x02   # May raise an exception
F_SIDE_EFFECT = 0x04  # Has side effects (writes state)
F_PURE_STACK = 0x08   # Pure stack manipulation (no side effects, no raise)
F_CONST_READ = 0x10   # Reads a constant (no raise, no side effect)

# Numeric opcode IDs for the IR — mapped from CPython opcode names.
# We assign stable IDs for the base opcodes plus our pseudo-ops.
# Using a dict for O(1) lookup in both directions.

_OPNAME_TO_ID = {}
_ID_TO_OPNAME = {}
_next_op_id = 0


def _register_op(name):
    """Register an opname and return its numeric ID."""
    global _next_op_id
    if name in _OPNAME_TO_ID:
        return _OPNAME_TO_ID[name]
    oid = _next_op_id
    _OPNAME_TO_ID[name] = oid
    _ID_TO_OPNAME[oid] = name
    _next_op_id += 1
    return oid


# Register all CPython opcodes
for _name in _opcode_mod.opname:
    if _name.startswith('<'):
        continue
    _register_op(_name)

# Register superinstruction pseudo-ops (Phase 4)
SUPER_LOAD_FAST_LOAD_FAST = _register_op('LOAD_FAST__LOAD_FAST')
SUPER_LOAD_FAST_LOAD_CONST = _register_op('LOAD_FAST__LOAD_CONST')
SUPER_LOAD_CONST_LOAD_FAST = _register_op('LOAD_CONST__LOAD_FAST')
SUPER_LOAD_FAST_STORE_FAST = _register_op('LOAD_FAST__STORE_FAST')
SUPER_STORE_FAST_LOAD_FAST = _register_op('STORE_FAST__LOAD_FAST')
SUPER_STORE_FAST_STORE_FAST = _register_op('STORE_FAST__STORE_FAST')
SUPER_LOAD_FAST_LOAD_FAST_BINARY_ADD = _register_op('LOAD_FAST__LOAD_FAST__BINARY_ADD')
SUPER_LOAD_FAST_LOAD_CONST_COMPARE = _register_op('LOAD_FAST__LOAD_CONST__COMPARE')
SUPER_LOAD_FAST_LOAD_ATTR = _register_op('LOAD_FAST__LOAD_ATTR')
SUPER_LOAD_FAST_BINARY_SUBSCR = _register_op('LOAD_FAST__BINARY_SUBSCR')

# Common opcode IDs (cached for hot-path use)
OP_NOP = _OPNAME_TO_ID.get('NOP', _register_op('NOP'))
OP_RESUME = _OPNAME_TO_ID.get('RESUME', _register_op('RESUME'))
OP_PRECALL = _OPNAME_TO_ID.get('PRECALL', _register_op('PRECALL'))
OP_LOAD_CONST = _OPNAME_TO_ID.get('LOAD_CONST', _register_op('LOAD_CONST'))
OP_LOAD_FAST = _OPNAME_TO_ID.get('LOAD_FAST', _register_op('LOAD_FAST'))
OP_STORE_FAST = _OPNAME_TO_ID.get('STORE_FAST', _register_op('STORE_FAST'))
OP_POP_TOP = _OPNAME_TO_ID.get('POP_TOP', _register_op('POP_TOP'))
OP_PUSH_NULL = _OPNAME_TO_ID.get('PUSH_NULL', _register_op('PUSH_NULL'))
OP_SWAP = _OPNAME_TO_ID.get('SWAP', _register_op('SWAP'))
OP_COPY = _OPNAME_TO_ID.get('COPY', _register_op('COPY'))
OP_BINARY_OP = _OPNAME_TO_ID.get('BINARY_OP', _register_op('BINARY_OP'))
OP_COMPARE_OP = _OPNAME_TO_ID.get('COMPARE_OP', _register_op('COMPARE_OP'))
OP_BINARY_SUBSCR = _OPNAME_TO_ID.get('BINARY_SUBSCR', _register_op('BINARY_SUBSCR'))
OP_LOAD_ATTR = _OPNAME_TO_ID.get('LOAD_ATTR', _register_op('LOAD_ATTR'))
OP_RETURN_VALUE = _OPNAME_TO_ID.get('RETURN_VALUE', _register_op('RETURN_VALUE'))
OP_JUMP_FORWARD = _OPNAME_TO_ID.get('JUMP_FORWARD', _register_op('JUMP_FORWARD'))
OP_JUMP_BACKWARD = _OPNAME_TO_ID.get('JUMP_BACKWARD', _register_op('JUMP_BACKWARD'))
OP_JUMP_BACKWARD_NO_INTERRUPT = _OPNAME_TO_ID.get('JUMP_BACKWARD_NO_INTERRUPT',
                                                    _register_op('JUMP_BACKWARD_NO_INTERRUPT'))
OP_JUMP_ABSOLUTE = _OPNAME_TO_ID.get('JUMP_ABSOLUTE', _register_op('JUMP_ABSOLUTE'))


def op_id(name):
    """Get numeric opcode ID for a name, registering if needed."""
    return _OPNAME_TO_ID.get(name) or _register_op(name)


def op_name(oid):
    """Get opcode name from numeric ID."""
    return _ID_TO_OPNAME.get(oid, '<unknown:%d>' % oid)


# ---------------------------------------------------------------------------
# Instruction tuple: (op_id, arg, offset, flags, jump_target, extra)
#   op_id:       int — numeric opcode
#   arg:         int — instruction argument
#   offset:      int — original byte offset (for debugging/exception correlation)
#   flags:       int — bitfield of F_* flags
#   jump_target: int|None — target offset for jump instructions
#   extra:       int — extra arg for superinstructions (second operand arg)
# ---------------------------------------------------------------------------
# For performance, we use plain tuples rather than namedtuples.
# Field indices:
IR_OP = 0
IR_ARG = 1
IR_OFFSET = 2
IR_FLAGS = 3
IR_JUMP = 4
IR_EXTRA = 5

# Tuple size
IR_SIZE = 6


def make_ir(op_id, arg, offset, flags=0, jump_target=None, extra=0):
    """Create a canonical IR instruction tuple."""
    return (op_id, arg, offset, flags, jump_target, extra)


# ---------------------------------------------------------------------------
# Specialized → base opcode mapping (CPython 3.11+ adaptive interpreter)
# ---------------------------------------------------------------------------
_SPECIALIZED_TO_BASE = {}
try:
    _base_opnames = set(_opcode_mod.opname)
    for _spec in getattr(_opcode_mod, '_specialized_instructions', []):
        _best = None
        for _base in _base_opnames:
            if _spec.startswith(_base) and (not _best or len(_base) > len(_best)):
                _best = _base
        if _best:
            _SPECIALIZED_TO_BASE[_spec] = _best
except Exception:
    pass


# ---------------------------------------------------------------------------
# Phase 2: Side-effect and may-raise classification
# ---------------------------------------------------------------------------

# Sets of opcodes by behavior category
_PURE_STACK_OPS = frozenset([
    'NOP', 'RESUME', 'PRECALL', 'PUSH_NULL', 'POP_TOP', 'SWAP', 'COPY',
    'ROT_TWO', 'ROT_THREE', 'ROT_FOUR', 'DUP_TOP', 'COPY_FREE_VARS',
])

_CONST_READ_OPS = frozenset([
    'LOAD_CONST', 'LOAD_FAST',
])

_WRITE_OPS = frozenset([
    'STORE_FAST', 'STORE_NAME', 'STORE_GLOBAL', 'STORE_ATTR', 'STORE_DEREF',
    'STORE_SUBSCR', 'DELETE_FAST', 'DELETE_NAME', 'DELETE_GLOBAL',
    'DELETE_ATTR', 'DELETE_DEREF', 'DELETE_SUBSCR',
    'LIST_APPEND', 'LIST_EXTEND', 'SET_ADD', 'SET_UPDATE', 'MAP_ADD',
    'DICT_UPDATE', 'DICT_MERGE', 'IMPORT_STAR',
])

_JUMP_OPS = frozenset([
    'JUMP_FORWARD', 'JUMP_BACKWARD', 'JUMP_BACKWARD_NO_INTERRUPT',
    'JUMP_ABSOLUTE',
    'POP_JUMP_FORWARD_IF_FALSE', 'POP_JUMP_FORWARD_IF_TRUE',
    'POP_JUMP_BACKWARD_IF_FALSE', 'POP_JUMP_BACKWARD_IF_TRUE',
    'POP_JUMP_FORWARD_IF_NONE', 'POP_JUMP_FORWARD_IF_NOT_NONE',
    'POP_JUMP_BACKWARD_IF_NONE', 'POP_JUMP_BACKWARD_IF_NOT_NONE',
    'POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE',
    'JUMP_IF_TRUE_OR_POP', 'JUMP_IF_FALSE_OR_POP',
    'FOR_ITER', 'SEND',
])

_UNCONDITIONAL_JUMP_OPS = frozenset([
    'JUMP_FORWARD', 'JUMP_BACKWARD', 'JUMP_BACKWARD_NO_INTERRUPT',
    'JUMP_ABSOLUTE',
])

_CONDITIONAL_JUMP_OPS = _JUMP_OPS - _UNCONDITIONAL_JUMP_OPS

_BLOCK_TERMINATOR_OPS = frozenset([
    'RETURN_VALUE', 'RAISE_VARARGS', 'RERAISE',
])
# Note: RETURN_GENERATOR is NOT a block terminator — the generator body
# (everything after RETURN_GENERATOR) is executed when the generator is
# iterated.  YIELD_VALUE returns from _run_frames but the frame continues
# on the next send()/next(), so it is also not a terminator.

# Pure binary ops (safe for constant folding): only immutable types, no metamethods
_PURE_BINARY_OPS = frozenset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# 0=+, 1=&, 2=//, 3=<<, 4=@(matmul), 5=*, 6=%, 7=|, 8=**, 9=>>, 10=-, 11=/, 12=^
# We only fold: + - * // % ** / & | ^ >> <<  (not matmul @)
_FOLDABLE_BINARY_OPS = frozenset([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12])
_FOLDABLE_TYPES = (int, float, bool)


def _classify_flags(opname_str):
    """Return flags bitfield for an opname."""
    flags = 0
    if opname_str in _PURE_STACK_OPS:
        flags |= F_PURE_STACK
    elif opname_str in _CONST_READ_OPS:
        flags |= F_CONST_READ
    else:
        # Default: may raise
        flags |= F_MAY_RAISE
    if opname_str in _WRITE_OPS:
        flags |= F_SIDE_EFFECT
    if opname_str in _JUMP_OPS:
        flags |= F_JUMP
    if opname_str in _BLOCK_TERMINATOR_OPS:
        flags |= F_SIDE_EFFECT  # block terminators have side effects
    return flags


# Precompute flags for all registered opcodes
_OP_FLAGS = {}
for _n, _oid in _OPNAME_TO_ID.items():
    _OP_FLAGS[_oid] = _classify_flags(_n)


# ---------------------------------------------------------------------------
# Stage 1: Raw decode (canonicalize CPython bytecode to IR)
# ---------------------------------------------------------------------------

def _raw_decode(code):
    """Decode a code object into canonical IR instructions.

    Returns (ir_instructions, offset_to_index, argvals, exc_table).
    - ir_instructions: list of IR tuples (op_id, arg, offset, flags, jump_target, extra)
    - offset_to_index: dict mapping byte offset -> instruction index
    - argvals: list of resolved argval per instruction (for jump targets)
    - exc_table: list of (start, end, target, depth, lasti) tuples
    """
    raw = list(_dis_mod.get_instructions(code, adaptive=True, show_caches=True))

    ir_instructions = []
    offset_to_index = {}
    argvals = []
    pending_extended = 0

    for instr in raw:
        # Skip CACHE pseudo-instructions
        if instr.opname == 'CACHE':
            continue

        # Fold EXTENDED_ARG if it appears
        if instr.opname == 'EXTENDED_ARG':
            pending_extended = (pending_extended | (instr.arg if instr.arg is not None else 0)) << 8
            # Still record the offset mapping so jumps targeting this location work
            offset_to_index[instr.offset] = len(ir_instructions)
            continue

        idx = len(ir_instructions)
        offset_to_index[instr.offset] = idx

        # Map specialized to base opname
        opname_str = _SPECIALIZED_TO_BASE.get(instr.opname, instr.opname)

        # Resolve argument (fold EXTENDED_ARG)
        arg = instr.arg if instr.arg is not None else 0
        if pending_extended:
            arg = pending_extended | arg
            pending_extended = 0

        # Get numeric op ID
        oid = _OPNAME_TO_ID.get(opname_str)
        if oid is None:
            oid = _register_op(opname_str)
            _OP_FLAGS[oid] = _classify_flags(opname_str)

        # Determine flags
        flags = _OP_FLAGS.get(oid, F_MAY_RAISE)

        # Determine jump target
        jump_target = None
        if flags & F_JUMP:
            jump_target = instr.argval

        ir_instructions.append(make_ir(oid, arg, instr.offset, flags, jump_target))
        argvals.append(instr.argval)

    # Parse exception table
    exc_table = []
    if hasattr(code, 'co_exceptiontable') and code.co_exceptiontable:
        try:
            for entry in _dis_mod._parse_exception_table(code):
                exc_table.append((entry.start, entry.end, entry.target, entry.depth,
                                  getattr(entry, 'lasti', False)))
        except Exception:
            pass

    return (ir_instructions, offset_to_index, argvals, exc_table)


# ---------------------------------------------------------------------------
# Phase 2: CFG construction
# ---------------------------------------------------------------------------

class BasicBlock:
    """A basic block in the control flow graph."""
    __slots__ = ('start', 'end', 'succs', 'preds', 'is_entry',
                 'is_exc_handler', 'reachable')

    def __init__(self, start, end=None):
        self.start = start       # index of first instruction (inclusive)
        self.end = end           # index of last instruction (exclusive)
        self.succs = []          # successor block start indices
        self.preds = []          # predecessor block start indices
        self.is_entry = False
        self.is_exc_handler = False
        self.reachable = False


def build_cfg(ir_instructions, offset_to_index, exc_table):
    """Build a control flow graph from IR instructions.

    Returns dict mapping block_start_index -> BasicBlock.
    """
    n = len(ir_instructions)
    if n == 0:
        return {}

    # Step 1: Identify leaders (block start indices)
    leaders = set()
    leaders.add(0)  # Entry point is always a leader

    for i, instr in enumerate(ir_instructions):
        flags = instr[IR_FLAGS]

        if flags & F_JUMP:
            jump_target = instr[IR_JUMP]
            if jump_target is not None and jump_target in offset_to_index:
                target_idx = offset_to_index[jump_target]
                leaders.add(target_idx)
            # Fallthrough of conditional jumps is a leader
            opname_str = op_name(instr[IR_OP])
            if opname_str not in _UNCONDITIONAL_JUMP_OPS:
                if i + 1 < n:
                    leaders.add(i + 1)
            else:
                # After unconditional jump, next instruction starts a new block
                if i + 1 < n:
                    leaders.add(i + 1)

        # Block terminators
        opname_str = op_name(instr[IR_OP])
        if opname_str in _BLOCK_TERMINATOR_OPS:
            if i + 1 < n:
                leaders.add(i + 1)

    # Exception handler targets are leaders
    for (start, end, target, depth, lasti) in exc_table:
        if target in offset_to_index:
            target_idx = offset_to_index[target]
            leaders.add(target_idx)

    # Step 2: Create blocks
    sorted_leaders = sorted(leaders)
    blocks = {}

    for li, leader in enumerate(sorted_leaders):
        end = sorted_leaders[li + 1] if li + 1 < len(sorted_leaders) else n
        block = BasicBlock(leader, end)
        if leader == 0:
            block.is_entry = True
        blocks[leader] = block

    # Mark exception handler blocks
    for (start, end, target, depth, lasti) in exc_table:
        if target in offset_to_index:
            target_idx = offset_to_index[target]
            if target_idx in blocks:
                blocks[target_idx].is_exc_handler = True

    # Step 3: Add edges
    for leader, block in blocks.items():
        if block.end <= block.start:
            continue

        last_idx = block.end - 1
        last_instr = ir_instructions[last_idx]
        last_flags = last_instr[IR_FLAGS]
        last_opname = op_name(last_instr[IR_OP])

        if last_flags & F_JUMP:
            jump_target = last_instr[IR_JUMP]
            if jump_target is not None and jump_target in offset_to_index:
                target_idx = offset_to_index[jump_target]
                if target_idx in blocks:
                    block.succs.append(target_idx)
                    blocks[target_idx].preds.append(leader)

            # Conditional jumps also fall through
            if last_opname not in _UNCONDITIONAL_JUMP_OPS:
                if block.end in blocks:
                    block.succs.append(block.end)
                    blocks[block.end].preds.append(leader)
        elif last_opname not in _BLOCK_TERMINATOR_OPS:
            # Fallthrough to next block
            if block.end in blocks:
                block.succs.append(block.end)
                blocks[block.end].preds.append(leader)

    # Step 4: Add exception edges
    for (start, end, target, depth, lasti) in exc_table:
        if target not in offset_to_index:
            continue
        target_idx = offset_to_index[target]
        if target_idx not in blocks:
            continue
        # Find all blocks whose instructions fall within [start, end)
        for leader, block in blocks.items():
            if block.end <= block.start:
                continue
            # Check if any instruction in this block has an offset in [start, end)
            first_offset = ir_instructions[block.start][IR_OFFSET]
            last_offset = ir_instructions[block.end - 1][IR_OFFSET]
            if first_offset < end and last_offset >= start:
                if target_idx not in block.succs:
                    block.succs.append(target_idx)
                if leader not in blocks[target_idx].preds:
                    blocks[target_idx].preds.append(leader)

    # Step 5: Mark reachable blocks (BFS from entry)
    worklist = [0]
    while worklist:
        idx = worklist.pop()
        if idx not in blocks or blocks[idx].reachable:
            continue
        blocks[idx].reachable = True
        for succ in blocks[idx].succs:
            if succ in blocks and not blocks[succ].reachable:
                worklist.append(succ)

    # Also mark exception handler targets as reachable
    for (start, end, target, depth, lasti) in exc_table:
        if target in offset_to_index:
            target_idx = offset_to_index[target]
            if target_idx in blocks and not blocks[target_idx].reachable:
                blocks[target_idx].reachable = True
                worklist = [target_idx]
                while worklist:
                    idx = worklist.pop()
                    if idx not in blocks:
                        continue
                    for succ in blocks[idx].succs:
                        if succ in blocks and not blocks[succ].reachable:
                            blocks[succ].reachable = True
                            worklist.append(succ)

    return blocks


# ---------------------------------------------------------------------------
# Phase 2: Stack effect validation
# ---------------------------------------------------------------------------

def validate_stack_effects(ir_instructions, blocks, offset_to_index):
    """Validate stack heights at basic block boundaries.

    Returns (valid, errors) where valid is bool and errors is list of strings.
    """
    if not blocks:
        return True, []

    errors = []
    # Compute stack height for each block entry
    stack_heights = {}  # block_start -> expected stack height
    stack_heights[0] = 0  # Entry block starts with empty stack

    worklist = [0]
    visited = set()

    while worklist:
        block_start = worklist.pop(0)
        if block_start in visited:
            continue
        if block_start not in blocks:
            continue
        visited.add(block_start)
        block = blocks[block_start]

        height = stack_heights.get(block_start, 0)

        # Walk through instructions in the block
        for i in range(block.start, block.end):
            if i >= len(ir_instructions):
                break
            instr = ir_instructions[i]
            oid = instr[IR_OP]
            arg = instr[IR_ARG]

            # Try to compute stack effect
            try:
                name = op_name(oid)
                # Skip pseudo-ops for stack effect computation
                if name.startswith('LOAD_FAST__') or name.startswith('STORE_FAST__'):
                    continue
                opc = _opcode_mod.opmap.get(name)
                if opc is not None:
                    effect = _dis_mod.stack_effect(opc, arg)
                    height += effect
            except (ValueError, KeyError):
                pass  # Unknown opcode, skip validation

        # Propagate to successors
        for succ in block.succs:
            if succ in stack_heights:
                # Already seen: check consistency
                if stack_heights[succ] != height and succ not in visited:
                    # Only warn, don't hard-fail (exception handlers reset stack)
                    if not blocks.get(succ, BasicBlock(0)).is_exc_handler:
                        errors.append(
                            'Stack height mismatch at block %d: expected %d, got %d'
                            % (succ, stack_heights[succ], height))
            else:
                stack_heights[succ] = height
            if succ not in visited:
                worklist.append(succ)

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Phase 3: Peephole optimizations
# ---------------------------------------------------------------------------

def _peephole_constant_fold(ir_instructions, code):
    """Fold LOAD_CONST a; LOAD_CONST b; BINARY_OP <pure> -> LOAD_CONST result.

    Only folds immutable numeric types (int, float, bool).
    Returns (modified_instructions, changed).
    """
    consts = code.co_consts
    result = list(ir_instructions)
    changed = False

    i = 0
    while i < len(result) - 2:
        instr_a = result[i]
        instr_b = result[i + 1]
        instr_c = result[i + 2]

        # Pattern: LOAD_CONST, LOAD_CONST, BINARY_OP
        if (instr_a[IR_OP] == OP_LOAD_CONST and
                instr_b[IR_OP] == OP_LOAD_CONST and
                instr_c[IR_OP] == OP_BINARY_OP):

            arg_a = instr_a[IR_ARG]
            arg_b = instr_b[IR_ARG]
            bin_op = instr_c[IR_ARG]

            # Only fold known pure ops on immutable types
            if bin_op in _FOLDABLE_BINARY_OPS:
                try:
                    val_a = consts[arg_a]
                    val_b = consts[arg_b]
                except (IndexError, TypeError):
                    i += 1
                    continue

                if (isinstance(val_a, _FOLDABLE_TYPES) and
                        isinstance(val_b, _FOLDABLE_TYPES)):
                    try:
                        # Compute the folded result
                        base = bin_op
                        if base == 0:  folded = val_a + val_b
                        elif base == 1:  folded = val_a & val_b
                        elif base == 2:  folded = val_a // val_b
                        elif base == 3:  folded = val_a << val_b
                        elif base == 5:  folded = val_a * val_b
                        elif base == 6:  folded = val_a % val_b
                        elif base == 7:  folded = val_a | val_b
                        elif base == 8:  folded = val_a ** val_b
                        elif base == 9:  folded = val_a >> val_b
                        elif base == 10: folded = val_a - val_b
                        elif base == 11: folded = val_a / val_b
                        elif base == 12: folded = val_a ^ val_b
                        else:
                            i += 1
                            continue

                        # Check for overflow or unreasonable results
                        if isinstance(folded, int) and folded.bit_length() > 1024:
                            i += 1
                            continue

                        # Find or add the folded constant
                        const_idx = None
                        for ci, cv in enumerate(consts):
                            if cv == folded and type(cv) is type(folded):
                                const_idx = ci
                                break

                        if const_idx is not None:
                            # Replace with LOAD_CONST folded; NOP; NOP
                            result[i] = make_ir(OP_LOAD_CONST, const_idx,
                                                instr_a[IR_OFFSET], F_CONST_READ)
                            result[i + 1] = make_ir(OP_NOP, 0,
                                                    instr_b[IR_OFFSET], F_PURE_STACK)
                            result[i + 2] = make_ir(OP_NOP, 0,
                                                    instr_c[IR_OFFSET], F_PURE_STACK)
                            changed = True
                            i += 3
                            continue
                    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
                        pass
        i += 1

    return result, changed


def _peephole_swap_cancel(ir_instructions):
    """Cancel adjacent SWAP(i); SWAP(i) pairs.

    Returns (modified_instructions, changed).
    """
    result = list(ir_instructions)
    changed = False

    i = 0
    while i < len(result) - 1:
        a = result[i]
        b = result[i + 1]
        if (a[IR_OP] == OP_SWAP and b[IR_OP] == OP_SWAP and
                a[IR_ARG] == b[IR_ARG]):
            result[i] = make_ir(OP_NOP, 0, a[IR_OFFSET], F_PURE_STACK)
            result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
            changed = True
            i += 2
        else:
            i += 1

    return result, changed


def _peephole_nop_removal(ir_instructions, offset_to_index):
    """Remove NOP instructions and rebuild offset_to_index.

    Returns (new_instructions, new_offset_to_index, changed).
    """
    new_instructions = []
    new_offset_to_index = {}
    index_map = {}  # old_index -> new_index
    changed = False

    for old_idx, instr in enumerate(ir_instructions):
        if instr[IR_OP] == OP_NOP:
            # Map this offset to the next real instruction
            new_offset_to_index[instr[IR_OFFSET]] = len(new_instructions)
            index_map[old_idx] = len(new_instructions)
            changed = True
            continue
        new_idx = len(new_instructions)
        new_offset_to_index[instr[IR_OFFSET]] = new_idx
        index_map[old_idx] = new_idx
        new_instructions.append(instr)

    if not changed:
        return ir_instructions, offset_to_index, False

    return new_instructions, new_offset_to_index, True


# ---------------------------------------------------------------------------
# Phase 3: CFG optimizations
# ---------------------------------------------------------------------------

def _jump_threading(ir_instructions, offset_to_index, blocks):
    """Thread jumps: if a jump target is itself an unconditional jump, collapse.

    Returns (modified_instructions, changed).
    """
    result = list(ir_instructions)
    changed = False

    for i, instr in enumerate(result):
        if not (instr[IR_FLAGS] & F_JUMP):
            continue
        jump_target = instr[IR_JUMP]
        if jump_target is None or jump_target not in offset_to_index:
            continue

        # Follow the chain of unconditional jumps (max depth to prevent loops)
        final_target = jump_target
        seen = set()
        for _ in range(10):  # max chain length
            if final_target in seen:
                break
            seen.add(final_target)
            target_idx = offset_to_index.get(final_target)
            if target_idx is None or target_idx >= len(result):
                break
            target_instr = result[target_idx]
            target_opname = op_name(target_instr[IR_OP])
            if target_opname in _UNCONDITIONAL_JUMP_OPS and target_instr[IR_JUMP] is not None:
                final_target = target_instr[IR_JUMP]
            else:
                break

        if final_target != jump_target:
            result[i] = make_ir(instr[IR_OP], instr[IR_ARG], instr[IR_OFFSET],
                                instr[IR_FLAGS], final_target, instr[IR_EXTRA])
            changed = True

    return result, changed


def _dead_block_elimination(ir_instructions, blocks):
    """Replace unreachable blocks with NOPs.

    Returns (modified_instructions, changed).
    """
    result = list(ir_instructions)
    changed = False

    for leader, block in blocks.items():
        if block.reachable or block.is_exc_handler:
            continue
        # Replace all instructions in unreachable block with NOP
        for i in range(block.start, min(block.end, len(result))):
            if result[i][IR_OP] != OP_NOP:
                result[i] = make_ir(OP_NOP, 0, result[i][IR_OFFSET], F_PURE_STACK)
                changed = True

    return result, changed


# ---------------------------------------------------------------------------
# Phase 4: Superinstruction fusion
# ---------------------------------------------------------------------------

def _fuse_superinstructions(ir_instructions, offset_to_index, blocks):
    """Fuse common instruction sequences into superinstructions.

    Only fuses within basic blocks and never across jump targets.
    Returns (modified_instructions, changed).
    """
    result = list(ir_instructions)
    changed = False

    # Build set of all jump target indices (no fusion across these)
    jump_targets = set()
    for instr in ir_instructions:
        if instr[IR_FLAGS] & F_JUMP and instr[IR_JUMP] is not None:
            target = offset_to_index.get(instr[IR_JUMP])
            if target is not None:
                jump_targets.add(target)
    # Exception handler targets
    for leader, block in blocks.items():
        if block.is_exc_handler:
            jump_targets.add(leader)

    # Process each basic block
    for leader, block in blocks.items():
        if not block.reachable:
            continue

        i = block.start
        while i < block.end - 1:
            # Don't fuse if next instruction is a jump target
            if i + 1 in jump_targets:
                i += 1
                continue

            a = result[i]
            b = result[i + 1]

            # Pattern: LOAD_FAST; LOAD_FAST
            if a[IR_OP] == OP_LOAD_FAST and b[IR_OP] == OP_LOAD_FAST:
                result[i] = make_ir(SUPER_LOAD_FAST_LOAD_FAST,
                                    a[IR_ARG], a[IR_OFFSET],
                                    F_CONST_READ, None, b[IR_ARG])
                result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                changed = True
                i += 2
                continue

            # Pattern: LOAD_FAST; LOAD_CONST
            if a[IR_OP] == OP_LOAD_FAST and b[IR_OP] == OP_LOAD_CONST:
                result[i] = make_ir(SUPER_LOAD_FAST_LOAD_CONST,
                                    a[IR_ARG], a[IR_OFFSET],
                                    F_CONST_READ, None, b[IR_ARG])
                result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                changed = True
                i += 2
                continue

            # Pattern: LOAD_CONST; LOAD_FAST
            if a[IR_OP] == OP_LOAD_CONST and b[IR_OP] == OP_LOAD_FAST:
                result[i] = make_ir(SUPER_LOAD_CONST_LOAD_FAST,
                                    a[IR_ARG], a[IR_OFFSET],
                                    F_CONST_READ, None, b[IR_ARG])
                result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                changed = True
                i += 2
                continue

            # Pattern: LOAD_FAST; STORE_FAST
            if a[IR_OP] == OP_LOAD_FAST and b[IR_OP] == OP_STORE_FAST:
                result[i] = make_ir(SUPER_LOAD_FAST_STORE_FAST,
                                    a[IR_ARG], a[IR_OFFSET],
                                    F_SIDE_EFFECT, None, b[IR_ARG])
                result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                changed = True
                i += 2
                continue

            # Pattern: STORE_FAST; LOAD_FAST
            if a[IR_OP] == OP_STORE_FAST and b[IR_OP] == OP_LOAD_FAST:
                result[i] = make_ir(SUPER_STORE_FAST_LOAD_FAST,
                                    a[IR_ARG], a[IR_OFFSET],
                                    F_SIDE_EFFECT, None, b[IR_ARG])
                result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                changed = True
                i += 2
                continue

            # Pattern: STORE_FAST; STORE_FAST
            if a[IR_OP] == OP_STORE_FAST and b[IR_OP] == OP_STORE_FAST:
                result[i] = make_ir(SUPER_STORE_FAST_STORE_FAST,
                                    a[IR_ARG], a[IR_OFFSET],
                                    F_SIDE_EFFECT, None, b[IR_ARG])
                result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                changed = True
                i += 2
                continue

            # Pattern: LOAD_FAST; LOAD_ATTR
            if a[IR_OP] == OP_LOAD_FAST and b[IR_OP] == OP_LOAD_ATTR:
                result[i] = make_ir(SUPER_LOAD_FAST_LOAD_ATTR,
                                    a[IR_ARG], a[IR_OFFSET],
                                    F_MAY_RAISE, None, b[IR_ARG])
                result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                changed = True
                i += 2
                continue

            # Pattern: LOAD_FAST; BINARY_SUBSCR
            if a[IR_OP] == OP_LOAD_FAST and b[IR_OP] == OP_BINARY_SUBSCR:
                result[i] = make_ir(SUPER_LOAD_FAST_BINARY_SUBSCR,
                                    a[IR_ARG], a[IR_OFFSET],
                                    F_MAY_RAISE, None, b[IR_ARG])
                result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                changed = True
                i += 2
                continue

            # 3-op pattern: LOAD_FAST; LOAD_FAST; BINARY_OP(ADD)
            if (i + 2 < block.end and i + 2 not in jump_targets):
                c = result[i + 2]
                if (a[IR_OP] == OP_LOAD_FAST and b[IR_OP] == OP_LOAD_FAST and
                        c[IR_OP] == OP_BINARY_OP and c[IR_ARG] == 0):  # ADD=0
                    result[i] = make_ir(SUPER_LOAD_FAST_LOAD_FAST_BINARY_ADD,
                                        a[IR_ARG], a[IR_OFFSET],
                                        F_MAY_RAISE, None, b[IR_ARG])
                    result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                    result[i + 2] = make_ir(OP_NOP, 0, c[IR_OFFSET], F_PURE_STACK)
                    changed = True
                    i += 3
                    continue

                # 3-op: LOAD_FAST; LOAD_CONST; COMPARE_OP
                if (a[IR_OP] == OP_LOAD_FAST and b[IR_OP] == OP_LOAD_CONST and
                        c[IR_OP] == OP_COMPARE_OP):
                    result[i] = make_ir(SUPER_LOAD_FAST_LOAD_CONST_COMPARE,
                                        a[IR_ARG], a[IR_OFFSET],
                                        F_MAY_RAISE, None,
                                        (b[IR_ARG] << 16) | c[IR_ARG])
                    result[i + 1] = make_ir(OP_NOP, 0, b[IR_OFFSET], F_PURE_STACK)
                    result[i + 2] = make_ir(OP_NOP, 0, c[IR_OFFSET], F_PURE_STACK)
                    changed = True
                    i += 3
                    continue

            i += 1

    return result, changed


# ---------------------------------------------------------------------------
# Main optimization pipeline
# ---------------------------------------------------------------------------

# Optimization levels:
#   0 = no optimization (raw IR, still uses numeric opcodes)
#   1 = peephole + CFG (constant fold, swap cancel, jump threading, dead blocks)
#   2 = level 1 + superinstruction fusion

def optimize(ir_instructions, offset_to_index, argvals, exc_table, code,
             opt_level=2):
    """Run the optimization pipeline on canonical IR.

    Returns (opt_instructions, opt_offset_to_index, opt_argvals).
    """
    if opt_level <= 0:
        return ir_instructions, offset_to_index, argvals

    # --- Level 1: Peephole and CFG optimizations ---
    instrs = ir_instructions
    o2i = offset_to_index

    # Constant folding
    instrs, _ = _peephole_constant_fold(instrs, code)

    # Swap cancellation
    instrs, _ = _peephole_swap_cancel(instrs)

    # Build CFG for CFG-level optimizations
    blocks = build_cfg(instrs, o2i, exc_table)

    # Jump threading
    instrs, _ = _jump_threading(instrs, o2i, blocks)

    # Dead block elimination
    instrs, _ = _dead_block_elimination(instrs, blocks)

    # --- Level 2: Superinstruction fusion ---
    if opt_level >= 2:
        # Rebuild CFG after peephole changes
        blocks = build_cfg(instrs, o2i, exc_table)
        instrs, _ = _fuse_superinstructions(instrs, o2i, blocks)

    # Remove NOPs and rebuild offset_to_index
    instrs, o2i, _ = _peephole_nop_removal(instrs, o2i)

    # Rebuild argvals to match the new instruction list
    # We need to map from old offsets to argvals
    old_argval_by_offset = {}
    for old_idx, instr in enumerate(ir_instructions):
        if old_idx < len(argvals):
            old_argval_by_offset[instr[IR_OFFSET]] = argvals[old_idx]

    new_argvals = []
    for instr in instrs:
        av = old_argval_by_offset.get(instr[IR_OFFSET])
        # For instructions with modified jump targets, update argval
        if instr[IR_JUMP] is not None:
            av = instr[IR_JUMP]
        new_argvals.append(av)

    return instrs, o2i, new_argvals


# ---------------------------------------------------------------------------
# Two-tier cache
# ---------------------------------------------------------------------------

# Tier 1: Decode cache (canonical IR, keyed by code object)
_DECODE_CACHE = _weakref_mod.WeakKeyDictionary()
_DECODE_HITS = 0
_DECODE_MISSES = 0

# Tier 2: Optimize cache (optimized IR, keyed by (code_obj, opt_level, py_minor))
_OPTIMIZE_CACHE = _weakref_mod.WeakKeyDictionary()
_OPTIMIZE_HITS = 0
_OPTIMIZE_MISSES = 0

_PY_MINOR = _sys_mod.version_info[:2]


def decode_cached(code):
    """Return cached canonical IR: (ir_instructions, offset_to_index, argvals, exc_table)."""
    global _DECODE_HITS, _DECODE_MISSES
    hit = _DECODE_CACHE.get(code)
    if hit is not None:
        _DECODE_HITS += 1
        return hit
    _DECODE_MISSES += 1
    payload = _raw_decode(code)
    _DECODE_CACHE[code] = payload
    return payload


def optimize_cached(code, opt_level=2):
    """Return cached optimized IR: (opt_instructions, opt_offset_to_index, opt_argvals, exc_table).

    Uses a two-tier cache:
      1. Decode cache: canonical IR (shared across optimization levels)
      2. Optimize cache: optimized IR per (code, opt_level, python_version)
    """
    global _OPTIMIZE_HITS, _OPTIMIZE_MISSES

    cache_key = (opt_level, _PY_MINOR)
    per_code = _OPTIMIZE_CACHE.get(code)
    if per_code is not None:
        hit = per_code.get(cache_key)
        if hit is not None:
            _OPTIMIZE_HITS += 1
            return hit

    _OPTIMIZE_MISSES += 1

    # Stage 1: Get canonical IR (from decode cache)
    ir_instructions, offset_to_index, argvals, exc_table = decode_cached(code)

    # Stage 2: Optimize
    opt_instrs, opt_o2i, opt_argvals = optimize(
        ir_instructions, offset_to_index, argvals, exc_table, code,
        opt_level=opt_level)

    result = (opt_instrs, opt_o2i, opt_argvals, exc_table)

    if per_code is None:
        per_code = {}
        _OPTIMIZE_CACHE[code] = per_code
    per_code[cache_key] = result

    return result


def decode_cache_stats():
    """Return (hits, misses, current_size) for decode cache."""
    return (_DECODE_HITS, _DECODE_MISSES, len(_DECODE_CACHE))


def optimize_cache_stats():
    """Return (hits, misses, current_size) for optimize cache."""
    return (_OPTIMIZE_HITS, _OPTIMIZE_MISSES, len(_OPTIMIZE_CACHE))
