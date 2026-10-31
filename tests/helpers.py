"""Tiny BBC-style corpus used by the tests."""
from pathlib import Path

CORPUS = {
    "business/001.txt": ("Shares fall as markets react",
                         "Stock markets fell sharply on Monday as investors sold shares after weak company profits. "
                         "Analysts said the market decline reflects fears about economic growth."),
    "business/002.txt": ("Bank raises interest rates",
                         "The central bank raised interest rates to fight inflation. Mortgage lenders warned that "
                         "higher rates will hurt borrowers and slow the economy."),
    "business/003.txt": ("Takeover bid for retailer",
                         "Shareholders will vote on a takeover bid for the high street retailer. The merger could "
                         "create the largest company in the sector."),
    "sport/001.txt": ("Chelsea win thrilling match",
                      "Chelsea scored twice in the second half to win the football match. The goal from the "
                      "captain sealed the victory at Stamford Bridge."),
    "sport/002.txt": ("England clinch cricket series",
                      "England won the final cricket test and the series. The bowlers took ten wickets as the "
                      "match ended early on day four."),
    "sport/003.txt": ("Olympic champion retires",
                      "The Olympic gold medal winner announced her retirement from athletics after a long and "
                      "successful career on the track."),
    "tech/001.txt": ("Mobile phones get smarter",
                     "New mobile phone technology lets users browse the internet and play video games. Broadband "
                     "access is also growing quickly."),
    "tech/002.txt": ("Virus spreads by email",
                     "A new computer virus is spreading through email attachments. Security experts urged users "
                     "to update their software and avoid spam."),
    "tech/003.txt": ("Digital music downloads rise",
                     "Sales of digital music downloads doubled as more people bought portable players. "
                     "Online music stores report record growth."),
}


def make_corpus(root: Path) -> Path:
    """Write the sample corpus under root/bbc and return that folder."""
    bbc = Path(root) / "bbc"
    for rel, (title, body) in CORPUS.items():
        path = bbc / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{title}\n\n{body}\n", encoding="utf-8")
    return bbc
