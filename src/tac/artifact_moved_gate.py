import ast
import re
import tokenize


def violations(path):
    source = path.read_text()
    tree, env, safe, found = ast.parse(source), {}, set(), []
    owner = re.match(r'ddm_[a-z0-9]+', path.stem)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'tac.artifact_moved':
            safe.update(a.asname or a.name for a in node.names if a.name == 'resolve')
        if isinstance(node, ast.Import):
            safe.update((a.asname or a.name) + suffix for a in node.names for name, suffix in (('tac.artifact_moved', '.resolve'), ('ddm_jg2_tail_reencode', '.resolve_artifact')) if a.name == name)
    comments = {t.start[0]: t.string for t in tokenize.generate_tokens(iter(source.splitlines(True)).__next__)
                if t.type == tokenize.COMMENT}
    def values(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return {n.value}
        if isinstance(n, ast.Name):
            return env.get(n.id, set())
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.Add)):
            return {a + ('/' if isinstance(n.op, ast.Div) else '') + b for a in values(n.left) for b in values(n.right)}
        if isinstance(n, ast.Call) and ast.unparse(n.func) not in safe:
            if ast.unparse(n.func) in {'Path', 'str', 'pathlib.Path'} and n.args:
                return values(n.args[0])
            if isinstance(n.func, ast.Attribute) and n.func.attr in {'resolve', 'absolute'}:
                return values(n.func.value)
            return set().union(*(values(a) for a in n.args))
        return set()
    class Scan(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            prior = dict(env)
            for arg in [*n.args.posonlyargs, *n.args.args, *n.args.kwonlyargs]:
                env.pop(arg.arg, None)
            for child in n.body:
                self.visit(child)
            env.clear()
            env.update(prior)
        def visit_Assign(self, n):
            self.generic_visit(n)
            for target in n.targets:
                if isinstance(target, ast.Name):
                    env[target.id] = values(n.value)
        def visit_Call(self, n):
            name = n.func.attr if isinstance(n.func, ast.Attribute) else ast.unparse(n.func)
            target = n.func.value if isinstance(n.func, ast.Attribute) and name in {'stat', 'open'} else (n.args[0] if n.args else None)
            modes = [*n.args, *(k.value for k in n.keywords if k.arg == 'mode')]
            write = name == 'open' and any(isinstance(a, ast.Constant) and isinstance(a.value, str) and re.fullmatch(r'[wax][bt+]*', a.value) for a in modes)
            waiver = comments.get(n.lineno, '').partition('MOVED_READ_OK:')[2].strip()
            waived = len(waiver) >= 8 and not any(s in waiver.lower() for s in ('<', 'todo', 'placeholder', 'rationale'))
            arms = set().union(*(set(re.findall(r'/Volumes/[^/]+/pact/(ddm_[a-z0-9]+)(?:_|/|$)', v)) for v in values(target))) - {owner}
            if name in {'stat', 'open', 'load'} and arms and not write and not waived:
                found.append(f'{path.name}:{n.lineno}: sibling artifact read bypasses tac.artifact_moved.resolve; MOVED_READ_OK:<specific explanation>; CLAUDE.md certify-or-block / ddm_mv1')
            self.generic_visit(n)
    scanner = Scan()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    env[target.id] = values(n.value)
    scanner.visit(tree)
    return found
