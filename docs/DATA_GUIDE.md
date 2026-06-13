# World Cup Data Guide (for people who don't watch football)

This explains **every file and every column** in plain English, plus just enough
football to understand the data. Read Part 1 once, then use Parts 2–3 as a reference.

---

## Part 1 - Football in 5 minutes (only what the data needs)

- **National team**: a country's team (e.g. "Brazil"). Players normally play for a
  **club** (a city team you pay to watch, e.g. Real Madrid) most of the year, and
  occasionally gather to play for their **country**. The data has both ideas.
- **The World Cup**: a tournament held **every 4 years** where national teams
  compete. The next one is **2026**.
- **Qualifiers vs the Finals**: hundreds of countries can't all play, so over the
  prior ~2 years they play **qualifying matches** to earn a spot. The ~32–48 that
  succeed go to the **"Finals"** - the actual month-long World Cup event. *(In this
  data, "World Cup matches" = the Finals; "qualifiers" are stored separately.)*
- **A match**: two teams play for 90 minutes. Each **goal** = 1 point toward that
  match. Most goals wins. A tie is a **draw**.
- **Home vs Away**: every match labels one team "home" and one "away". Normally
  "home" means it's played in that team's country (an advantage - local crowd). In
  a neutral tournament it's just a label for which side of the record is which.
- **Two stages of the World Cup**:
  1. **Group stage**: teams are split into small **groups**. Everyone plays everyone
     in their group once. **Win = 3 points, draw = 1, loss = 0.** Top 2 advance.
  2. **Knockout stage**: lose and you're out. Round of 16 → Quarter-final →
     Semi-final → Final. The Final winner is the **champion**.
- **What if a knockout match is a draw?** It can't stay tied, so:
  - **Extra time**: 30 extra minutes.
  - **Penalty shootout**: if still tied, players take turns shooting 1-on-1 at the
    goalkeeper. This is a *tiebreaker*, separate from the normal score.
- **FIFA Ranking**: FIFA (the governing body) keeps a year-round **ranking of all
  ~210 national teams** by a points system, updated monthly from every match they
  play. Rank 1 = best. It exists independently of the World Cup and is the main way
  to gauge "how strong is this team right now."

**Football jargon you'll see in columns** (full glossary in Part 3): *goal, own
goal, penalty, yellow/red card, substitute, cap, xG, manager, captain, attendance.*

---

## Part 2 - The files, and what each column means

There are **two data sources** in `data/`:

- **`kaggle_2026/`** - the richer, up-to-date set (includes 2022 + the 2026 schedule
  + FIFA rankings). This is the **backbone**.
- **`maven_analytics/`** - broader history (every international match since 1872) +
  the 2022 squads. Fills in context the other set lacks.

> ⚠️ The two sources spell some countries differently (e.g. `Iran` vs `IR Iran`).
> The code in `src/teams.py` fixes this automatically - always load data through
> `src/load.py`, not by reading the CSVs raw.

### 📁 kaggle_2026/

#### `fifa_ranking_2026-06-08.csv` - team strength just before the 2026 World Cup
One row per national team (211 teams). A snapshot of the FIFA ranking on 8 Jun 2026.

| Column | Plain English |
|---|---|
| `team` | Country name |
| `team_code` | 3-letter code (BRA = Brazil) |
| `association` | Continental confederation (UEFA = Europe, CONMEBOL = South America, etc.) |
| `rank` | Current world rank (1 = best) |
| `previous_rank` | Their rank in the previous monthly update |
| `points` | FIFA points (higher = stronger). This is the real strength measure. |
| `previous_points` | Points in the previous update (lets you see if rising/falling) |
| `rated_matches` | How many recent matches fed into this rating |

> There's also `fifa_ranking_2022-10-06.csv` - same columns (no `rated_matches`). It's
> the snapshot before the **2022** World Cup, so you can test "does ranking predict results?"

#### `matches_1930_2022.csv` - every World Cup Finals match ever, in rich detail
One row per match (964), from the 1930 up to the 2022 World Cup. **44 columns** - the
most detailed file you have. The core columns:

| Column | Plain English |
|---|---|
| `home_team`, `away_team` | The two countries |
| `home_score`, `away_score` | Goals each scored (this decides the winner) |
| `home_xg`, `away_xg` | **Expected Goals** - a quality stat (see glossary). *Only exists for recent matches.* |
| `home_penalty`, `away_penalty` | Goals in a penalty **shootout** tiebreaker, if there was one |
| `home_manager`, `away_manager` | The coach of each team |
| `home_captain`, `away_captain` | The team's on-field leader |
| `Attendance` | How many people watched in the stadium |
| `Venue` | Stadium and city |
| `Officials` | Referees and assistants |
| `Round` | Stage (Group stage, Round of 16, Final, etc.) |
| `Date`, `Year` | When it was played |
| `Score` | The score as text (e.g. "(4) 3–3 (2)" - brackets are the shootout) |
| `Referee` | Main referee |
| `Notes` | Free-text notes (e.g. "won on penalties") |
| `Host` | The country hosting that World Cup |

The remaining ~22 columns (ending in `_long`, plus `_goal`, `_card`, etc.) are
**event detail** - lists of *who* did *what* and *when*. They're mostly empty for
old matches and are stringified lists, e.g. `home_goal` = `"Messi · 108"` means Messi
scored in the 108th minute. Examples:

| Column group | What it records |
|---|---|
| `home_goal`, `home_goal_long` | Who scored for the home team (short / detailed) |
| `home_own_goal` | A player accidentally scoring into their **own** net (counts for the opponent) |
| `home_penalty_goal` | Goals scored from a penalty kick *during* the match |
| `home_penalty_shootout_goal_long` / `_miss_long` | Shootout attempts made / missed |
| `home_red_card` | Player sent off (ejected) - see glossary |
| `home_yellow_card_long` | Players cautioned |
| `home_substitute_in_long` | Players who came on as substitutes |

*(Each has an `away_` twin. You can ignore most of these unless you want player-level detail.)*

#### `schedule_2026.csv` - the upcoming 2026 World Cup fixtures
One row per scheduled match (72 - **group stage only** so far; knockouts get added
once teams qualify). The `Score` column is **empty on purpose** - these haven't been
played yet (the tournament starts 11 Jun 2026).

| Column | Plain English |
|---|---|
| `Round` | Stage (all "Group stage" for now) |
| `Day`, `Date`, `Time` | When it's played |
| `Score` | Result - **blank until the match happens** |
| `Referee`, `Notes` | Blank until played |
| `Year` | 2026 |
| `home_team`, `away_team` | The two countries scheduled |

#### `world_cup.csv` - one-line summary of every World Cup (1930–2022)
One row per tournament (22).

| Column | Plain English |
|---|---|
| `Year` | Which World Cup |
| `Host` | Host country |
| `Teams` | How many teams took part |
| `Champion`, `Runner-Up` | 1st and 2nd place |
| `TopScorrer` | Player who scored most goals (name + count). *(sic - column is misspelled in the file)* |
| `Attendance`, `AttendanceAvg` | Total / average crowd |
| `Matches` | Matches played |

### 📁 maven_analytics/

#### `international_matches.csv` - EVERY international match since 1872 (the big one)
One row per match (17,769). This is **not** World Cup Finals - it's friendlies,
qualifiers, and continental cups. Your source for "recent form."

| Column | Plain English |
|---|---|
| `ID` | Row identifier |
| `Tournament` | What competition it was ("Friendly", "FIFA World Cup qualification", "Copa America", etc.) |
| `Date` | When |
| `Home Team`, `Away Team` | The two countries |
| `Home Goals`, `Away Goals` | Goals each scored |
| `Win Conditions` | Note if decided by penalties |
| `Home Stadium` | TRUE if played in the home team's country |

#### `2022_world_cup_squads.csv` - the players at the 2022 World Cup
One row per player (831). The only **player-level** file.

| Column | Plain English |
|---|---|
| `ID` | Player identifier |
| `Team` | Country they play for |
| `Position` | Goalkeeper / Defender / Midfielder / Forward |
| `Player` | Player name *(some have a "(captain)" suffix - minor dirty data)* |
| `Age` | Age in 2022 |
| `Caps` | Total matches ever played for their country (a **cap** = one appearance). High = experienced. |
| `Goals` | Total career goals **for their country** (all matches, not just World Cup) |
| `WC Goals` | Of those, how many at World Cups |
| `League` | Country of the league their club is in |
| `Club` | Their club team |

#### `world_cup_matches.csv` - historical WC Finals results (simpler than Kaggle's)
One row per match (900, **1930–2018 - no 2022**). Overlaps with the Kaggle match file
but with fewer columns. Use Kaggle's instead unless you need this format.

| Column | Plain English |
|---|---|
| `ID`, `Year`, `Date` | Identifier and when |
| `Stage` | Group stage / knockout round |
| `Home Team`, `Away Team` | The two countries |
| `Home Goals`, `Away Goals` | Goals each |
| `Win Conditions` | Decided in extra time / penalties / "golden goal" (an old rule) |
| `Host Team` | TRUE if the host country is the home team here |

#### `world_cups.csv` - tournament summary (Maven's version)
Same idea as Kaggle's `world_cup.csv` but with 3rd/4th place. **The 2022 row is blank**
(missing data) - prefer Kaggle's `world_cup.csv` for 2022.

| Column | Plain English |
|---|---|
| `Year`, `Host Country` | Which cup, where |
| `Winner`, `Runners-Up`, `Third`, `Fourth` | Top 4 finishers |
| `Goals Scored` | Total goals in the tournament |
| `Qualified Teams` | How many teams took part |
| `Matches Played` | Matches in the tournament |

#### `2022_world_cup_matches.csv` - the 2022 fixture list (64 matches)
| Column | Plain English |
|---|---|
| `ID`, `Year`, `Date` | Identifier and when |
| `Stage` | Group stage / knockout round |
| `Home Team`, `Away Team` | The two countries (in knockouts these may read as "Winner of Match X") |
| `Host Team` | TRUE if Qatar (the host) is the home team |

#### `2022_world_cup_groups.csv` - who was in which group (32 rows)
| Column | Plain English |
|---|---|
| `Group` | Group letter A–H. Teams in a group all play each other; top 2 advance. |
| `Team` | Country |
| `FIFA Ranking` | Their rank at the time |

#### `data_dictionary.csv`
Maven's own column definitions (the source for several explanations above). A
reference file, not data to analyze.

---

## Part 3 - Glossary of every football term in the data

| Term | Meaning |
|---|---|
| **Goal** | The ball going into the net. 1 goal = 1 toward the match score. Most goals wins. |
| **Score** | Goals for each side, e.g. "2–1" (home 2, away 1). |
| **Draw / tie** | Equal score. Allowed in group stage; in knockouts it goes to extra time / penalties. |
| **xG (Expected Goals)** | A modern quality stat: how many goals a team *should* have scored given the chances they created. 2.5 xG but only 1 goal = unlucky/wasteful. Only recorded for recent matches. |
| **Cap** | One appearance for your national team. "80 caps" = played 80 times for your country. A measure of experience. |
| **Penalty (penalty kick)** | A free 1-on-1 shot at the goal awarded for a foul. Easy to score. Counts as a normal goal. |
| **Penalty shootout** | A *tiebreaker* after a drawn knockout match: teams alternate penalty kicks; most scored wins. Tracked separately from the match score. |
| **Own goal** | A player accidentally puts the ball into *their own* net - the point goes to the opponent. |
| **Yellow card** | A warning for a foul. |
| **Red card** | Sent off - that player leaves and can't be replaced (team plays with 10). Two yellows = a red ("yellow_red_card"). |
| **Substitute** | A fresh player swapped in for a tired/injured one during the match. |
| **Manager** | The coach who picks the team and tactics. |
| **Captain** | The team's designated on-field leader (also wears the armband). |
| **Attendance** | Number of spectators in the stadium. |
| **Home / Away** | Labels for the two sides. "Home" usually means it's that team's country (an advantage). |
| **Group stage** | First phase: round-robin within small groups; top 2 advance. Win 3 pts, draw 1, loss 0. |
| **Knockout stage** | Single-elimination: lose and you're out. Round of 16 → Quarter → Semi → Final. |
| **Qualifier** | A match played *before* the World Cup to earn a spot in it. |
| **Friendly** | A practice match that doesn't count toward any competition. |
| **Confederation / association** | Regional football body (UEFA = Europe, CONMEBOL = South America, CONCACAF = North/Central America, CAF = Africa, AFC = Asia, OFC = Oceania). |
| **FIFA points / rank** | The year-round strength score and ranking of national teams. Rank 1 = best. |

---

## TL;DR - which file do I use for what?

| I want… | Use |
|---|---|
| How strong is each team going into 2026? | `kaggle_2026/fifa_ranking_2026-06-08.csv` |
| The 2026 match schedule (to predict) | `kaggle_2026/schedule_2026.csv` |
| Detailed past World Cup results | `kaggle_2026/matches_1930_2022.csv` |
| Recent team form (all matches, any year) | `maven_analytics/international_matches.csv` |
| Player info for 2022 | `maven_analytics/2022_world_cup_squads.csv` |
| Past tournament winners | `kaggle_2026/world_cup.csv` |

Load everything through `src/load.py` so country names line up across sources.
