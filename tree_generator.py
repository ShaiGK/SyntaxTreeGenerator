import itertools
from collections import defaultdict
from functools import lru_cache

from nltk import Tree

import data.data as data

# Node type: either a plain symbol string, or a tuple ('GROUP', optional:bool, children:list[node_or_string])
Node = str | tuple[str, bool, list['Node']]  # ('GROUP', optional, children)


def split_top_level(s: str, sep: str = '|') -> list[str]:
    """
    Splits a string at the top level based on a given separator, while preserving the nested structures
    within parentheses. This function ensures that separators inside parentheses are not treated as
    delimiter points. The returned list contains parts of the string split by the separator at the
    topmost level.

    :param s: The input string to be split.
    :param sep: The separator used to delimit the top-level split points in the string. Defaults to '|'.
    :return: A list of strings split at the top level by the separator.
    :raises ValueError: If there are unmatched opening or closing parentheses in the input string.
    """

    parts = []
    buf = []
    level = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '(':
            level += 1
            buf.append(ch)
        elif ch == ')':
            level -= 1
            if level < 0:
                raise ValueError("Unmatched closing parenthesis in RHS: " + s)
            buf.append(ch)
        elif ch == sep and level == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if level != 0:
        raise ValueError("Unmatched opening parenthesis in RHS: " + s)
    last = ''.join(buf).strip()
    if last:
        parts.append(last)

    return parts


def tokenize_alternative(s: str) -> list[Node]:
    """
    Parses a string representation of alternatives into a structured list of nodes. The input string
    may include plain symbols, quoted terminals, and groupings denoted by parentheses.

    The function processes the input string iteratively and recursively:
    - Whitespace is ignored.
    - Plain symbols are extracted until encountering whitespace, parentheses, or specific delimiters.
    - Quoted terminals are parsed and added as tokens.
    - Parenthesized groups are identified and recursively processed to support nested levels.

    :param s: The input string representing alternatives, which may include plain symbols, quoted
        terminals, and parenthesized groupings.
    :return: A list of `Node` objects representing the structured alternatives parsed from the input
        string.
    :raises ValueError: If the input string contains unmatched parentheses, empty groups,
        unterminated quoted strings, or unexpected characters.
    """

    tokens: list[Node] = []
    i = 0
    n = len(s)
    while i < n:
        # skip whitespace
        if s[i].isspace():
            i += 1
            continue
        ch = s[i]
        if ch == '(':
            # parse group until matching ')'
            start = i + 1
            level = 1
            i = start
            while i < n and level > 0:
                if s[i] == '(':
                    level += 1
                elif s[i] == ')':
                    level -= 1
                i += 1
            if level != 0:
                raise ValueError(f"Unmatched '(' in alternative: {s}")
            # substring inside parentheses is s[start:i-1]
            inner = s[start:i - 1].strip()
            if inner == '':
                raise ValueError("Empty parentheses group are not allowed: '()'")
            # recursively tokenize inner content into child nodes
            # inner may contain top-level alternatives? Not here — parentheses only group symbols.
            child_nodes = tokenize_alternative(inner)
            tokens.append(('GROUP', True, child_nodes))
            continue  # i already positioned after ')'
        elif ch == '"':
            # quoted terminal
            j = i + 1
            esc = False
            buf = []
            while j < n:
                if esc:
                    buf.append(s[j])
                    esc = False
                elif s[j] == '\\':
                    esc = True
                elif s[j] == '"':
                    break
                else:
                    buf.append(s[j])
                j += 1
            if j >= n or s[j] != '"':
                raise ValueError(f"Unterminated quote in alternative: {s[i:]}")
            terminal = ''.join(buf)
            tokens.append(terminal)
            i = j + 1
        else:
            # plain symbol: read until whitespace or parentheses
            j = i
            buf = []
            while j < n and (not s[j].isspace()) and s[j] not in '()|':
                buf.append(s[j])
                j += 1
            sym = ''.join(buf)
            if sym == '':
                raise ValueError(f"Unexpected character at position {i} in: {s}")
            tokens.append(sym)
            i = j

    return tokens


def expand_nodes(nodes: list[Node]) -> list[list[str]]:
    """
    Expands a list of nodes, which may include strings and group nodes representing
    hierarchical structures, into all possible valid sequences. Handles both optional
    and required groups. Ensures the resulting sequences are valid and deduplicates
    the expansions while preserving their order.

    :param nodes: A list containing either strings or tuples representing group nodes.
                  Strings are treated as individual elements of the expansion. Tuples
                  represent groups, defined as ('GROUP', optional, children), where:
                  - 'optional' is a boolean indicating whether the group is optional.
                  - 'children' is a list of nodes representing the group's content.
    :return: A list of lists where each sublist is a unique, valid sequence formed by
             expanding the given nodes.
    :raises ValueError: If the expansion results in an explosion of more than 10,000
                        sequences for one alternative or if all expansions lead to
                        empty sequences, violating CNF/CYK requirements.
    """

    results: list[list[str]] = [[]]  # start with empty sequence

    for node in nodes:
        if isinstance(node, str):
            part_seqs = [[node]]
            new_results = []
            for seq in results:
                for part in part_seqs:
                    new_results.append(seq + part)
            results = new_results
        else:
            tag, optional, children = node
            assert tag == 'GROUP'
            included = expand_nodes(children)  # list of lists
            new_results = []
            if optional:
                # omit case: keep sequences as they are
                for seq in results:
                    new_results.append(list(seq))  # omit group
            # include case: append each included expansion to current sequences
            for seq in results:
                for inc in included:
                    new_results.append(seq + inc)
            results = new_results

        # safety: prevent explosion
        if len(results) > 10000:
            raise ValueError("Expansion exploded: more than 10,000 expansions for one alternative. "
                             "Refuse to expand; consider simplifying grammar.")
    # filter out empty expansions (epsilon) - not allowed for CYK
    filtered = [r for r in results if len(r) > 0]
    if not filtered:
        # all expansions empty -> invalid for CNF/CYK
        raise ValueError("An alternative expanded to only empty productions (epsilon), which is not supported by CYK.")
    # deduplicate while preserving order
    seen = set()
    unique = []
    for seq in filtered:
        tup = tuple(seq)
        if tup not in seen:
            seen.add(tup)
            unique.append(seq)

    return unique


def parse_rules(rules_string: str) -> dict[str, list[list[str]]]:
    """
    Parses a string representing a set of grammar rules and converts it into a dictionary
    where keys are non-terminal symbols (LHS) and values are lists of lists of symbols
    representing the corresponding production rules (RHS).

    Any lines that are empty or start with '#' as comments are ignored. Rules must follow
    the format: LHS -> RHS, with LHS specifying a non-terminal and RHS containing
    one or more valid alternatives. Each alternative can be a sequence of symbols.

    :param rules_string: A string containing grammar rules. Each rule must follow the
        format `LHS -> RHS`, where LHS is a non-terminal symbol and RHS is a sequence of
        alternatives separated by '|'. Symbols in RHS may include terminals or
        non-terminals, which are delimited by whitespace.
    :return: A dictionary mapping each non-terminal symbol (str) in the grammar to a list
        of alternative production rules. Each production rule is represented as a list
        of symbols (terminals and/or non-terminals).
    :raises ValueError: If a rule is malformed, such as missing a '->' in the rule, having
        an empty LHS, or any RHS alternative that is empty.
    """

    grammar: dict[str, list[list[str]]] = {}
    lines = rules_string.splitlines()
    for line_idx, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        # split on '->' first (only first occurrence)
        if '->' not in line:
            raise ValueError(f"Line {line_idx}: missing '->' in rule: {line}")
        lhs_part, rhs_part = line.split('->', 1)
        lhs = lhs_part.strip()
        if lhs == '':
            raise ValueError(f"Line {line_idx}: empty LHS in rule: {line}")
        # split top-level alternatives
        alternatives = split_top_level(rhs_part.strip(), sep='|')
        for alt in alternatives:
            alt = alt.strip()
            if alt == '':
                raise ValueError(f"Line {line_idx}: empty RHS alternative in rule: {line}")
            # tokenize alternative (handles nested parentheses)
            nodes = tokenize_alternative(alt)
            # expand into explicit symbol sequences
            expansions = expand_nodes(nodes)
            # convert quoted terminals: we already tokenized terminals as strings e.g. 'the' (if quoted)
            # but non-terminals are also strings; we keep them as is.
            for seq in expansions:
                grammar.setdefault(lhs, []).append(seq)

    return grammar


def convert_to_cnf(grammar: dict[str, list[list[str]]]) -> tuple[dict[str, list[list[str]]], set[str]]:
    """
    Converts a given context-free grammar (CFG) to Chomsky Normal Form (CNF).

    This function accepts a dictionary representation of a context-free grammar (CFG),
    where keys are non-terminal symbols and values are lists of production rules (themselves
    lists of symbols). The function modifies this grammar to adhere to Chomsky Normal Form
    (CNF), ensuring that each production rule conforms to one of the following patterns:
    - A -> BC, where A, B, and C are non-terminal symbols.
    - A -> a, where A is a non-terminal symbol and "a" is a terminal symbol.

    During the conversion process, the function introduces helper non-terminal symbols as
    necessary to ensure all rules fit CNF requirements. The function returns both the
    updated CNF-compliant grammar and a set of these newly introduced helper symbols.

    :return: A tuple containing:
        - The CNF-compliant grammar as a dictionary, where keys are non-terminal symbols,
          and values are lists of production rules (themselves lists of symbols).
        - A set of helper non-terminal symbols introduced during the conversion process.
    :param grammar: A dictionary representing the input context-free grammar. The dictionary's
        keys are strings corresponding to non-terminal symbols, and the values are lists of
        production rules, where each production rule is a list of strings.
    :rtype: tuple[dict[str, list[list[str]]], set[str]]
    """

    cnf_grammar = {}
    new_symbol_counter = itertools.count(1)
    helper_symbols = set()

    helper_cache = {}

    def add_rule(lhs: str, rhs: tuple[str, ...]) -> None:
        """
        Adds a production rule to the context-free grammar (CFG). The rule associates a
        left-hand side (LHS) symbol with its corresponding right-hand side (RHS) production.
        If the LHS symbol already exists in the grammar, the RHS production is appended
        to its existing rules.

        :param lhs: The left-hand side symbol of the production rule.
        :param rhs: The right-hand side symbols of the production rule, represented as a list of strings.
        :return: None
        """

        cnf_grammar.setdefault(lhs, set()).add(rhs)

    def get_helper_symbol(rhs: tuple[str, ...]) -> str:
        if rhs in helper_cache:
            return helper_cache[rhs]
        new_sym = f"X{next(new_symbol_counter)}"
        helper_cache[rhs] = new_sym
        helper_symbols.add(new_sym)
        return new_sym

    for lhs, productions in grammar.items():
        for rhs in productions:
            if len(rhs) <= 2:
                # Binary, Unary or terminal rule (keep as-is)
                add_rule(lhs, tuple(rhs))
                continue

            # Break down into binary rules with helper nodes
            last_pair = tuple(rhs[-2:])
            prev = get_helper_symbol(last_pair)
            add_rule(prev, last_pair)

            for i in range(len(rhs) - 3, 0, -1):
                segment = tuple(rhs[i:i + 1] + [prev])
                curr = get_helper_symbol(segment)
                add_rule(curr, (rhs[i], prev))
                prev = curr
            add_rule(lhs, (rhs[0], prev))

    return cnf_grammar, helper_symbols


def cyk_parse(tokens: list[str], grammar: dict[str, set[list[str]]]) -> list[list[dict[str, set]]]:
    """
    Parses a sequence of tokens using the CYK parsing algorithm with the provided
    context-free grammar. The algorithm builds a parse table that represents
    possible derivations of the input tokens from the grammar's start symbol.

    The CYK parser operates in three main phases:
    1. Initialization: Fills the diagonal of the parse table with preterminal rules
       derived from grammar terminal symbols.
    2. Unary Closure: Recursively applies unary grammar rules to extend possible
       derivations.
    3. Larger Spans: Combines smaller spans of parse table entries using binary
       grammar rules to fill larger spans.

    This implementation assumes that the grammar is in Chomsky Normal Form (CNF).
    In CNF, each production rule has either two nonterminals or one terminal
    on its right-hand side.

    :param tokens: List of terminal symbols (input sequence) to parse.
    :param grammar: Context-free grammar rules defined as a dictionary where keys
        are nonterminal symbols (strings) and values are sets of productions
        (each production is a list of terminal and/or nonterminal symbols).
    :return: A 2D table representing possible derivations. Each cell in the
        table is a dictionary mapping nonterminal symbols (strings) to a set
        of derivations (backpointers) for that span.
    """

    n = len(tokens)
    # Table[i][j] = dict mapping Nonterminal -> list of backpointers
    # span from i (inclusive) to j (exclusive)
    table = [[defaultdict(set) for _ in range(n + 1)] for _ in range(n)]

    # Precompute inverse grammar: RHS -> LHS rules
    inverse = defaultdict(set)
    for lhs, productions in grammar.items():
        for rhs in productions:
            inverse[rhs].add(lhs)

    def apply_unary_closure(cell: dict[str, set]) -> None:
        """
        Applies a unary closure operation on the given cell dictionary. The function modifies
        the dictionary in place by iterating through the keys in the input and appending
        new relationships based on predefined inverse mappings.

        :param cell: A dictionary where each key is a string and the associated value
            is a list of tuples indicating relationships for unary closure operation.
        :returns: None
        """

        key_copy = list(cell.keys())
        seen = set()
        for rhs in key_copy:
            if rhs in seen:
                continue
            for lhs in inverse.get((rhs,), []):
                cell[lhs].add(("unary", rhs))
                key_copy.append(lhs)
                seen.add(rhs)

    # Fill diagonal with preterminals
    for i, word in enumerate(tokens):
        for lhs in inverse.get((word,), []):
            table[i][i + 1][lhs].add(("terminal", word))
        apply_unary_closure(table[i][i + 1])

    # Fill larger spans
    for span in range(2, n + 1):  # span length
        for i in range(n - span + 1):  # start index
            j = i + span  # end index
            for k in range(i + 1, j):  # split point
                left_cell = table[i][k]
                right_cell = table[k][j]

                for B in left_cell:
                    for C in right_cell:
                        for lhs in inverse.get((B, C), []):
                            table[i][j][lhs].add(("binary", k, B, C))

            apply_unary_closure(table[i][j])

    return table


def extract_trees(table: list[list[dict[str, set]]],
                  tokens: list[str],
                  start_symbol: str = "S") -> tuple:
    """
    Returns (count, generator) where:
      - count is the exact number of parse trees (computed by DP)
      - generator yields parse trees lazily, one at a time

    The table format expected: table[i][j] maps symbol -> set of backpointer tuples,
    where backpointer is one of:
      ("terminal", word)
      ("unary", rhs_symbol)
      ("binary", split_k, B, C)
    """

    n = len(tokens)

    @lru_cache(maxsize=None)
    def count(symbol: str, i: int, j: int) -> int:
        if i < 0 or j > n or (i >= j and j != i + 1):
            return 0
        cell = table[i][j]
        if symbol not in cell:
            return 0
        total = 0
        for bp in cell[symbol]:
            kind = bp[0]
            if kind == "terminal":
                total += 1
            elif kind == "unary":
                rhs = bp[1]
                total += count(rhs, i, j)
            elif kind == "binary":
                _, k, B, C = bp
                total += count(B, i, k) * count(C, k, j)
            else:
                raise ValueError(f"Unknown backpointer kind: {kind}")
        return total

    total_trees = count(start_symbol, 0, n)

    # gen_state maps (symbol,i,j) -> dict with keys: cache(list), producer(generator or None), finished(bool)
    gen_state: dict[tuple[str, int, int], dict] = {}

    def node_producer(symbol: str, i: int, j: int):
        """
        Generator that produces parse trees for (symbol, i, j).
        It uses recursive calls to yield children via yield_from_node.
        """
        cell = table[i][j]
        backptrs = cell.get(symbol, ())
        for bp in backptrs:
            kind = bp[0]
            if kind == "terminal":
                word = bp[1]
                # yield a terminal-tree representation
                yield (symbol, word)

            elif kind == "unary":
                rhs = bp[1]
                # delegate to rhs's generator
                for child in yield_from_node(rhs, i, j):
                    yield (symbol, child)

            elif kind == "binary":
                _, k, B, C = bp
                # We need to produce cartesian product of lefts and rights lazily.
                # We'll iterate lefts; for each left we iterate rights, caching rights so they can be replayed.
                # Use the helper yield_from_node for both children.
                # We must be mindful that both left and right are themselves lazy caches.
                left_iter = yield_from_node(B, i, k)
                # iterate left trees as they are produced
                for left in left_iter:
                    # For each left, iterate right trees by calling yield_from_node.
                    # yield_from_node will return cached rights first and then produce more as needed.
                    for right in yield_from_node(C, k, j):
                        yield (symbol, left, right)
            else:
                raise ValueError(f"Unknown backpointer kind: {kind}")

    def yield_from_node(symbol: str, i: int, j: int):
        """
        Iterator that yields parse trees for (symbol,i,j), using incremental caching.
        """
        key = (symbol, i, j)
        if key not in gen_state:
            # initialize state and create producer generator (but do not start it)
            gen_state[key] = {'cache': [], 'producer': node_producer(symbol, i, j), 'finished': False}

        state = gen_state[key]

        # First yield from already cached results
        for item in state['cache']:
            yield item

        if state['finished']:
            return

        # Pull new items from the producer one at a time, append to cache, yield each.
        prod = state['producer']
        try:
            while True:
                next_item = next(prod)
                state['cache'].append(next_item)
                yield next_item
        except StopIteration:
            state['finished'] = True
            return

    # top-level generator that yields trees for start symbol
    def tree_generator():
        # If there are zero trees, return nothing
        if total_trees == 0:
            return
            yield  # keep this a generator
        for tree in yield_from_node(start_symbol, 0, n):
            yield tree

    return total_trees, tree_generator()


def format_tree(tree: tuple[str | tuple, ...], helper_symbols: set[str] | None = None) -> str:
    """
    Formats a given tree structure into a string representation suitable for human reading.

    The tree is represented as a tuple with the first element being the label and the subsequent
    elements being its children. If child nodes contain a specific symbol from `helper_symbols`,
    their grandchildren are recursively included to create a flat and simplified structure.
    This function ensures correct and compact conversion of a hierarchical structure into a
    string form.

    :param tree: Input tree structure represented as a tuple. The first element is the label,
        and subsequent elements represent its children, which can be strings or nested nodes.
    :param helper_symbols: A set of symbols to identify nodes requiring their grandchildren
        to be included in the output rather than their immediate children. Defaults to an
        empty set if not provided.
    :return: A string representation of the input tree structure.
    """

    if helper_symbols is None:
        helper_symbols = set()

    def gather_children(node: tuple[str | tuple, ...]) -> list[tuple[str | tuple, ...]]:
        """
        Gathers all children of a given node into a flat list. If a node's child contains
        a specific symbol (indicated by `helper_symbols`), its grandchildren are recursively
        flattened and included in the output list. The function ensures the hierarchical
        structure is simplified into a single level list.

        :param node: The node whose children are to be gathered and processed.
        :return: A flattened list of the children of the input node. If certain children
                 themselves are nodes, their hierarchical structure is recursively traversed.
        """

        raw_children = node[1:]
        out_children = []
        for child in raw_children:
            if isinstance(child, str):
                out_children.append(child)
            else:
                if child[0] in helper_symbols:
                    # flatten grandchildren
                    grandchildren = gather_children(child)
                    out_children.extend(grandchildren)
                else:
                    out_children.append(child)

        return out_children

    def node_to_str(node: tuple[str | tuple, ...]) -> str:
        """
        Convert a node represented as a tuple into a string format.

        The function transforms a node, which is a tuple-like structure with a label
        and its children, into a readable string representation. Each node contains a
        label as its first element and may have child nodes or strings. If a node
        contains exactly one child that is a string, it creates a compact output.
        Otherwise, it processes the children recursively to create a nested
        representation.

        :param node: The input tuple representing a node. The first element is
            its label, and the subsequent elements are its children, which
            can either be strings or nested nodes.
        :return: A string representation of the given node, including
            its label and the stringified children.
        """

        label = node[0]
        children = gather_children(node)

        if len(children) == 1 and isinstance(children[0], str):
            return f"[{label} {children[0]}]"

        child_strs = []
        for c in children:
            if isinstance(c, str):
                child_strs.append(c)
            else:
                child_strs.append(node_to_str(c))

        return f"[{label} " + " ".join(child_strs) + "]"

    return node_to_str(tree)


def display_parses(parse_trees: list[str], pretty_print: bool = True, draw: bool = True) -> None:
    """
    Displays parse trees provided in bracket notation, optionally pretty-printing them in text form
    and allowing interactive visualization as images.

    The function takes a list of parse trees (strings in bracket notation), converts them into NLTK
    Tree objects, and optionally pretty-prints and/or visualizes the trees. Users are prompted
    interactively if they want to view specific parses as graphical images.

    :param parse_trees: A list of parse trees represented as strings in bracket notation (e.g., "[S [NP ...]").
    :param pretty_print: Boolean flag indicating whether to pretty-print the trees in text form. Defaults to True.
    :param draw: Boolean flag indicating whether to enable interactive image visualization of the trees. Defaults to True.
    :return: None.
    """

    nltk_trees = []
    for i, parse_str in enumerate(parse_trees, 1):
        print(f"\nParse {i}:")
        print(parse_str)  # bracket notation

        # convert to nltk Tree
        t = Tree.fromstring(parse_str.replace("[", "(").replace("]", ")"))
        nltk_trees.append(t)

        if pretty_print:
            # pretty print
            print()
            t.pretty_print()

    if not nltk_trees:
        print("No successful parses.")
        return

    while draw:
        choice = input(
            f"\nEnter the number of a parse to view as an image (1-{len(nltk_trees)}), or 'no' to stop: "
        ).strip().lower()

        if choice == "no":
            break
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(nltk_trees):
                nltk_trees[idx - 1].draw()
            else:
                print("Invalid number.")
        else:
            print("Please enter a valid number or 'no'.")


def main(demo: bool = False, display: bool = True, pretty_print: bool = True, draw: bool = True,
         rules: str = None, sentence_list: list = None, pos_tags_list: list = None) -> None:
    """
    Parses a sentence using a context-free grammar (CFG), converts it to Chomsky Normal
    Form (CNF), and performs the CYK parsing algorithm to extract all possible parse trees
    for the input sentence. Additionally, it formats, cleans, and optionally draws the parse trees.

    :param demo: If True, prompts the user to provide a sentence and its corresponding
        part-of-speech (POS) tags. Defaults to False.
    :param display: If True, displays the parse trees in bracket notation and optionally
        allows users to view them interactively. Defaults to True.
    :param pretty_print: If True, formats the parse trees for better readability in output.
        Defaults to True.
    :param draw: If True, generates graphical visualizations of the parsed trees.
        Defaults to True.
    :param rules: An optional string containing the context-free grammar rules. Defaults to None.
    :param sentence_list: An optional list of sentences to parse. Defaults to None.
    :param pos_tags_list: An optional list of POS tags corresponding Îto the sentences. Defaults to None.
    :return: None
    """

    # Phrase structure rules
    rules = data.rules if rules is None else rules

    # Example sentence
    if demo:
        sentence_list = [input("Enter a sentence: ")]
        pos_tags_list = [input("Enter POS tags: ")]
        print()
    else:
        sentence_list = data.sentence_list if sentence_list is None else sentence_list
        pos_tags_list = data.pos_tags_list if pos_tags_list is None else pos_tags_list

    # Check there are the same number of sentences and POS tags.
    assert len(sentence_list) == len(pos_tags_list), "Please ensure same number of sentences and POS tags."

    print(f"Parsing {len(sentence_list)} {"sentence" if len(sentence_list) == 1 else "sentences"}:")

    parse_list: list[list[str]] = []

    for idx, (sentence, pos_tags) in enumerate(zip(sentence_list, pos_tags_list)):
        print()
        if sentence.startswith("*"):
            grammatical = False
            sentence = sentence[1:]
        else:
            grammatical = True
        print(f"Parsing ({"" if grammatical else "un"}grammatical) sentence number {idx + 1}: {sentence}")

        sentence = sentence.split()
        pos_tags = pos_tags.upper().split()
        new_rules = rules

        # Terminal additions to the PSRs
        assert len(sentence) == len(pos_tags), "Please ensure same length for sentence and pos_tags."
        rule_set = {f'{tag} -> "{word}"' for word, tag in zip(sentence, pos_tags)}
        new_rules += '\n' + '\n'.join(rule_set)

        # Parse rules and convert to CNF
        grammar = parse_rules(new_rules)
        print("Grammar created successfully.")
        cnf, helpers = convert_to_cnf(grammar)
        print("Grammar converted to CNF successfully.")

        # Run CYK
        table = cyk_parse(sentence, cnf)
        print("CYK parsing completed successfully.")

        # Extract bracketed parses
        count, gen = extract_trees(table, sentence, start_symbol="S")

        print("Parse extraction completed successfully.")
        if count <= 50:
            trees = list(gen)
        else:
            trees = []
            for _, t in zip(range(50), gen):
                trees.append(t)
            print(f"{count} trees found. There is no way I'm showing that... Here are the first 50 instead.")

        print(f"Found {count} parse(s):")
        expected = (grammatical and len(trees) > 0) or (not grammatical and len(trees) == 0)
        if not expected:
            print("THIS IS NOT EXPECTED.")

        # Format and clean parses for display and store them
        parse_strs = [format_tree(t, helper_symbols=helpers) for t in trees]
        parse_list.append(parse_strs)

        # Display results
        if display:
            display_parses(parse_strs, pretty_print=pretty_print, draw=draw)


if __name__ == "__main__":
    main(demo=False, display=True, draw=True)

    # TODO: Add support for {} choices
    # TODO: Possibly add support for N-N compounds
    # TODO: Add block testing option
    # TODO: Handle duplicate terminal rules
    # TODO: Split POS off of N
    # TODO: Add display choice after bulk run
    # TODO: Add summary after bulk run
