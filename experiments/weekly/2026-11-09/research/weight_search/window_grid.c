/* Independent cash-start windows with equal weights, rank buffer and exact fees. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HOLDINGS 10
#define WIDTH 13
#define FACTORS 9
#define SHORTLIST 32

typedef struct { double score; int id, sector; } Stock;

static int rank_order(const void *left, const void *right) {
    const Stock *a = left, *b = right;
    if (a->score != b->score) return a->score > b->score ? -1 : 1;
    return a->id - b->id;
}

static int contains(const int *ids, int count, int id) {
    for (int i = 0; i < count; ++i) if (ids[i] == id) return 1;
    return 0;
}

static void keep(Stock stock, int cap, int *sectors, int *ids, int *count) {
    if (sectors[stock.sector] < cap && !contains(ids, *count, stock.id)) {
        ++sectors[stock.sector];
        ids[(*count)++] = stock.id;
    }
}

/* Both retained and fresh names follow the current score order. */
static int targets(const Stock *ranks, int count, int cap, int sector_count,
                   int buffer, const int *previous, int *selected) {
    int sectors[sector_count];
    memset(sectors, 0, sizeof(sectors));
    int n = 0;
    if (buffer && previous) {
        for (int j = 0; j < 15 && j < count && n < HOLDINGS; ++j)
            if (contains(previous, HOLDINGS, ranks[j].id)) keep(ranks[j], cap, sectors, selected, &n);
    }
    for (int j = 0; j < count && n < HOLDINGS; ++j) keep(ranks[j], cap, sectors, selected, &n);
    return n == HOLDINGS;
}

static double trading_amount(const double *shared, int count, double capital, double invest) {
    double overlap = 0.0;
    for (int j = 0; j < count; ++j) overlap += fmin(shared[j], invest / HOLDINGS);
    return capital + invest - 2.0 * overlap;
}

/* Open-to-open ratios and close-to-open ratios are invariant to the number of shares. */
int evaluate_grid(const double *factors, const int *ids, const int *sectors, const int *counts,
                  int width, int weeks, int sector_count, int stocks, const double *marks,
                  const double *overnight, const int *days, const int *starts, int windows,
                  const double *weights, int cap, int buffer, double fee, double *out) {
    Stock *ranks = malloc((size_t)weeks * width * sizeof(Stock));
    int *lengths = malloc((size_t)weeks * sizeof(int));
    if (!ranks || !lengths) { free(ranks); free(lengths); return 1; }
    Stock all[width];
    for (int w = 0; w < weeks; ++w) {
        Stock *best = ranks + (size_t)w * width;
        int n = 0;
        for (int j = 0; j < counts[w]; ++j) {
            size_t p = (size_t)w * width + j;
            const double *x = factors + p * FACTORS;
            double score = 0.0;
            for (int k = 0; k < FACTORS; ++k) score += x[k] * weights[k];
            Stock stock = {score, ids[p], sectors[p]};
            all[j] = stock;
            if (n == SHORTLIST && rank_order(&stock, &best[n - 1]) >= 0) continue;
            int lo = 0, hi = n;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (rank_order(&stock, &best[mid]) < 0) hi = mid;
                else lo = mid + 1;
            }
            if (n < SHORTLIST) ++n;
            memmove(best + lo + 1, best + lo, (size_t)(n - lo - 1) * sizeof(Stock));
            best[lo] = stock;
        }
        int first[HOLDINGS];
        if (!targets(best, n, cap, sector_count, 0, NULL, first)) {
            n = counts[w];
            memcpy(best, all, (size_t)n * sizeof(Stock));
            qsort(best, (size_t)n, sizeof(Stock), rank_order);
        }
        lengths[w] = n;
    }

    for (int i = 0; i < windows; ++i) {
        int start = starts[i], previous[HOLDINGS];
        double invested = 1.0 / (1.0 + fee), peak = 1.0, dd = 0.0, nav = NAN, turnover = 0.0;
        int valid = 1;
        for (int w = start; w < start + WIDTH; ++w) {
            int picked[HOLDINGS];
            if (!targets(ranks + (size_t)w * width, lengths[w], cap, sector_count,
                         buffer, w == start ? NULL : previous, picked)) { valid = 0; break; }
            if (w > start) {
                const double *ratio = overnight + (size_t)w * stocks;
                double shared[HOLDINGS], capital = 0.0;
                int common = 0;
                for (int j = 0; j < HOLDINGS; ++j) {
                    double value = ratio[previous[j]] / HOLDINGS;
                    capital += value;
                    if (contains(picked, HOLDINGS, previous[j])) shared[common++] = value;
                }
                if (!(capital > 0.0) || !isfinite(capital)) { valid = 0; break; }
                double next = capital;
                for (int iteration = 0; iteration < 12; ++iteration) {
                    double update = capital - fee * trading_amount(shared, common, capital, next);
                    double difference = fabs(update - next);
                    next = update;
                    if (difference <= 1e-14) break;
                }
                turnover += trading_amount(shared, common, capital, next) / capital;
                invested *= next;
            }
            const double *daily = marks + (size_t)w * 5 * stocks;
            for (int d = 0; d < days[w]; ++d) {
                double value = 0.0;
                for (int j = 0; j < HOLDINGS; ++j) value += daily[(size_t)d * stocks + picked[j]];
                nav = invested * value / HOLDINGS;
                if (!(nav > 0.0) || !isfinite(nav)) { valid = 0; break; }
                peak = fmax(peak, nav);
                dd = fmax(dd, 1.0 - nav / peak);
            }
            if (!valid) break;
            memcpy(previous, picked, sizeof(previous));
        }
        out[3 * i] = valid ? nav - 1.0 : NAN;
        out[3 * i + 1] = valid ? dd : NAN;
        out[3 * i + 2] = valid ? turnover / (WIDTH - 1) : NAN;
    }
    free(lengths);
    free(ranks);
    return 0;
}
