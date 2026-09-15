"""Training rewards from replayed events; no evaluation diagnostic imports."""
from __future__ import annotations

import ast
from fractions import Fraction
from itertools import product
import json
import math

from agent.agent import AgentRun
from evaluation.verifier import VerificationResult
from grpo.schemas import RewardResult
from tasks.schemas import Task


REWARD_CONFIG = {"success":1.0, "selection_with_evidence":0.15, "legal_completion":0.02,
                 "invalid":-0.10, "redundant":-0.02, "no_final":-0.05, "efficiency":-0.02,
                 "count_cap":3, "length_denominator":16, "version":"evidence-shaping-v1"}
INVALID_CODES = {"UNKNOWN_TOOL","MISSING_ARGUMENT","INVALID_ARGUMENT","UNKNOWN_COMPANY","UNKNOWN_FIELD","INVALID_EXPRESSION"}


def _fraction(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError("Not a finite number")
    return Fraction(str(value))


def _margin_expression(expression: str, profit, revenue) -> bool:
    """Evidence of P/R computation, not just a coincidentally matching output.

    Numeric literals equal to observed P/R can be symbolic operands. Exact
    polynomial cross multiplication proves P/R, allowing P*(1/R), parentheses,
    unary signs and scaled ratios without a template or probabilistic probes.
    """
    try:
        tree = ast.parse(expression.strip(),mode="eval").body
        p, r = _fraction(profit), _fraction(revenue)
        if r == 0:
            return False
        literals = [n for n in ast.walk(tree) if isinstance(n,ast.Constant)]
        options = [["constant"] + (["p"] if _fraction(n.value) == p else [])
                   + (["r"] if _fraction(n.value) == r else []) for n in literals]
        if math.prod(len(o) for o in options) > 4096 or not any("p" in o for o in options) or not any("r" in o for o in options):
            return False

        def add(a,b):
            result = dict(a)
            for key,value in b.items():
                result[key] = result.get(key,0) + value
            return {key:value for key,value in result.items() if value}

        def multiply(a,b):
            result = {}
            for (x,y),v in a.items():
                for (u,w),z in b.items():
                    key = (x+u,y+w)
                    result[key] = result.get(key,0) + v*z
            return {key:value for key,value in result.items() if value}

        one = {(0,0):Fraction(1)}
        ps,rs = {(1,0):Fraction(1)},{(0,1):Fraction(1)}

        def evaluate(node, roles):
            if isinstance(node,ast.Constant):
                value = _fraction(node.value)
                role = roles[id(node)]
                return (ps if role == "p" else rs if role == "r" else {(0,0):value}),one
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
                a,b = evaluate(node.operand,roles)
                return (a if isinstance(node.op,ast.UAdd) else {k:-v for k,v in a.items()}),b
            if isinstance(node,ast.BinOp):
                a,b = evaluate(node.left,roles)
                c,d = evaluate(node.right,roles)
                if isinstance(node.op,ast.Add): return add(multiply(a,d),multiply(c,b)),multiply(b,d)
                if isinstance(node.op,ast.Sub): return add(multiply(a,d),{k:-v for k,v in multiply(c,b).items()}),multiply(b,d)
                if isinstance(node.op,ast.Mult): return multiply(a,c),multiply(b,d)
                if isinstance(node.op,ast.Div) and c: return multiply(a,d),multiply(b,c)
            raise ValueError("Unsupported arithmetic")

        for assignment in product(*options):
            if "p" not in assignment or "r" not in assignment:
                continue
            roles = {id(n):role for n,role in zip(literals,assignment,strict=True)}
            numerator,denominator = evaluate(tree,roles)
            if denominator and multiply(numerator,rs) == multiply(denominator,ps):
                return True
        return False
    except (ValueError,TypeError,SyntaxError,ZeroDivisionError,RecursionError,OverflowError):
        return False


def complete_l4_evidence(task: Task, run: AgentRun) -> bool:
    if task.difficulty != 4:
        return False
    candidates = set(task.metadata["companies"])
    observed, calculated = {}, set()
    for step in run.tool_steps:
        call,result = step.tool_call,step.tool_result
        if not result.success:
            continue
        if call.tool_name == "lookup_company":
            company,field = call.arguments.get("company"),call.arguments.get("field")
            if company in candidates and field in {"profit","revenue"}:
                observed[(company,field)] = result.output
        elif call.tool_name == "calculator":
            expression = call.arguments.get("expression")
            if not isinstance(expression,str):
                continue
            for company in candidates:
                p,r = observed.get((company,"profit")),observed.get((company,"revenue"))
                try:
                    matches = math.isclose(float(_fraction(result.output)),float(_fraction(p)/_fraction(r)),
                        rel_tol=task.verification_spec["rel_tol"],abs_tol=task.verification_spec["abs_tol"])
                except (ValueError,ZeroDivisionError,OverflowError):
                    continue
                if matches and _margin_expression(expression,p,r):
                    calculated.add(company)
    return calculated == candidates


def compute_reward(task: Task, run: AgentRun, verification: VerificationResult, variant: str) -> RewardResult:
    if variant not in {"outcome_only","shaped"}:
        raise ValueError("Unknown reward variant")
    if any(e.backend_error for e in run.events) or any(s.tool_result.error_code == "EXECUTION_ERROR" for s in run.tool_steps):
        raise RuntimeError("Infrastructure failure is not a policy reward")
    invalid = sum(e.parse_error is not None for e in run.events)
    seen, redundant = set(),0
    for step in run.tool_steps:
        invalid += step.tool_result.error_code in INVALID_CODES
        arguments = dict(step.tool_call.arguments)
        if isinstance(arguments.get("expression"),str):
            arguments["expression"] = arguments["expression"].strip()
        key = json.dumps([step.tool_call.tool_name,arguments],sort_keys=True,allow_nan=False)
        if step.tool_result.success:
            redundant += key in seen
            seen.add(key)
    success = int(verification.success)
    # Selection is deliberately separate from numeric/format success. S is
    # still solely the Verifier verdict; C follows its explicit evidence rule.
    answer = run.final_answer if run.has_final_answer else None
    company = answer.get("company") if isinstance(answer,dict) else None
    company = company.strip() if isinstance(company,str) else None
    evidence = int(complete_l4_evidence(task,run))
    selection = int(task.difficulty == 4 and company == task.ground_truth["company"] and evidence)
    legal = int(run.has_final_answer and invalid == 0 and all(s.tool_result.success for s in run.tool_steps))
    counts = {"success":success,"selection_with_evidence":selection,"complete_l4_evidence":evidence,
              "legal_completion":legal,"invalid":invalid,"redundant":redundant,
              "no_final":int(not run.has_final_answer),"tool_calls":len(run.tool_steps)}
    components = {"success":float(success)}
    if variant == "shaped":
        components.update(selection_with_evidence=0.15*selection,legal_completion=0.02*legal,
                          invalid=-0.1*min(invalid,3),redundant=-0.02*min(redundant,3),
                          no_final=-0.05*counts["no_final"],efficiency=-0.02*success*len(run.tool_steps)/16)
    return RewardResult(sum(components.values()),components,counts)
