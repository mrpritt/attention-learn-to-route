# Solomon VRPTW instances

This directory contains text extractions of the 56 Solomon VRPTW benchmark instances from archived copies of Solomon's original web pages.

## Format

Each file is a simple whitespace-separated table:

```text
# VEHICLE_CAPACITY <capacity>
CUST_NO XCOORD YCOORD DEMAND READY_TIME DUE_DATE SERVICE_TIME
...
```

The rows preserve the numbering used on the archived web pages: the first row is the depot, followed by 100 customer rows.

## Sources used

Archived Solomon pages used for extraction:

- C1 (`c101`--`c109`): <http://web.archive.org/web/20220619192735/http://web.cba.neu.edu/~msolomon/c101.htm>
- C2 (`c201`--`c208`): <http://web.archive.org/web/20211022011412/http://web.cba.neu.edu/~msolomon/c201.htm>
- R1 (`r101`--`r112`): <http://web.archive.org/web/20211021231016/http://web.cba.neu.edu/~msolomon/r101.htm>
- R2 (`r201`--`r211`): <http://web.archive.org/web/20211022001047/http://web.cba.neu.edu/~msolomon/r201.htm>
- RC1 (`rc101`--`rc108`): <http://web.archive.org/web/20211022011730/http://web.cba.neu.edu/~msolomon/rc101.htm>
- RC2 (`rc201`--`rc208`): <http://web.archive.org/web/20211021235323/http://web.cba.neu.edu/~msolomon/rc201.htm>

Extraction script:

- `../../scripts/fetch_solomon.py`

## 25- and 50-customer subsets

SINTEF's TOP VRPTW pages state that the smaller Solomon benchmark instances are formed by taking prefixes of the 100-customer instances:

- 25-customer page: <https://www.sintef.no/projectweb/top/vrptw/25-customers/>
  - states that the instances contain the **25 first customers** of the 100-customer instances.
- 50-customer page: <https://www.sintef.no/projectweb/top/vrptw/50-customers/>
  - states that the instances contain the **50 first customers** of the 100-customer instances.

Thus, to construct a 25- or 50-customer version from these files, keep the depot row plus the first 25 or 50 customer rows from the corresponding 100-customer instance.

## R-family coordinate reuse

In the extracted canonical files, all R-family instances (`r101`--`r112` and `r201`--`r211`) use the same 100 customer coordinates in the same order. They share depot `(35,35)`. The R1/R2 instances differ in horizon/capacity and time-window data, not in customer geography. By contrast, the C and RC families use different coordinate sets.

For our purposes this is mainly a useful fact for validation. We do not attempt to exactly reproduce Solomon's canonical coordinate set in the synthetic generator; the goal is a Solomon-like VRPTW distribution for training rather than bitwise regeneration of the historical benchmark files.
