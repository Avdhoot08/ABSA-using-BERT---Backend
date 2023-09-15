import "./app.scss";
import {
    Button,
    Card,
    Elevation,
    FileInput,
    Checkbox,
    Intent,
    H1,
    H2,
    TextArea,
    Dialog,
    DialogBody,
    DialogFooter,
    Icon,
} from "@blueprintjs/core";
import { parseCSVFile, getRandomElements } from "./utils";
import { useState } from "react";
import { submitReviews } from "./api";

function App() {
    const [files, setFiles] = useState(null);
    const [fileSelectorText, setFileSelectorText] = useState(
        "Choose a CSV file..."
    );
    const [reviews, setReviews] = useState([]);
    const [selectedReviews, setSelectedReviews] = useState({});
    const [reviewSelectMode, setReviewSelectMode] = useState(false);
    const [importDialogOpen, setImportDialogOpen] = useState(false);
    const [evaluation, setEvaluation] = useState(null);

    const toggleReviewSelectMode = () => {
        setReviewSelectMode((current) => !current);
    };

    const selectReview = ({ id, text, checked }) => {
        if (!checked) {
            const { [id]: foo, ...remaining } = selectedReviews;
            setSelectedReviews(remaining);
        } else {
            setSelectedReviews({ ...selectedReviews, [id]: text });
        }
    };

    const filesSelected = (event) => {
        setFiles(event.target.files);
        setFileSelectorText(event.target.files[0].name);
    };

    const getData = async () => {
        const reviews = await parseCSVFile(files[0]);
        setReviews(reviews);
        setFiles([]);
        setImportDialogOpen(false);
    };

    const selectRandomReviews = () => {
        const revs = getRandomElements(reviews, 30);
        const newRevs = {};
        revs.forEach(({ id, text }) => (newRevs[id] = text));
        setSelectedReviews({ ...selectedReviews, ...newRevs });
    };

    const submitReviewsForAnalysis = async () => {
        const payload = [];
        for (const [key, value] of Object.entries(selectedReviews)) {
            payload.push({ id: key, text: value });
        }
        const response = await submitReviews(payload);
        setEvaluation(response);
    };

    const footerActions = (
        <>
            <Button onClick={() => setImportDialogOpen(false)}>Cancel</Button>
            <Button intent={Intent.PRIMARY} disabled={!files} onClick={getData}>
                Import
            </Button>
        </>
    );

    const highlightReview = (review) => {
        let text = review.text;
        for (const { aspect, sentiment } of review.result) {
            text = text.replace(
                aspect,
                `<span class="review review--${sentiment.toLowerCase()}">${aspect}</span>`
            );
        }
        return text;
    };

    return (
        <main>
            <Dialog
                title="Import Data"
                icon="info-sign"
                isOpen={importDialogOpen}
                onClose={() => setImportDialogOpen(!importDialogOpen)}
            >
                <DialogBody>
                    <FileInput
                        text={fileSelectorText}
                        fill={true}
                        onInputChange={filesSelected}
                    />
                </DialogBody>
                <DialogFooter actions={footerActions} />
            </Dialog>

            <div id="board">
                <H1>Review Analyzer</H1>
                <TextArea className="review-input" growVertically={true} fill={true} />
                <Button>Analyze</Button>
                <div className="evaluation">
                    <H2>Results</H2>
                    <table class="bp4-html-table">
                        <thead>
                            <tr>
                                <th>Aspect</th>
                                <th>
                                    <Icon
                                        icon={"thumbs-up"}
                                        size={20}
                                        intent={Intent.SUCCESS}
                                    />
                                </th>
                                <th>
                                    <Icon
                                        icon={"thumbs-down"}
                                        size={20}
                                        intent={Intent.DANGER}
                                    />
                                </th>
                                <th>
                                    <Icon
                                        icon={"help"}
                                        size={20}
                                        intent={Intent.WARNING}
                                    />
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {evaluation &&
                                evaluation.aggregation &&
                                Object.entries(evaluation.aggregation).map(
                                    ([aspect, sentiments]) => {
                                        return (
                                            <tr>
                                                <td>{aspect}</td>
                                                <td>{sentiments["POS"]}</td>
                                                <td>{sentiments["NEG"]}</td>
                                                <td>{sentiments["NEU"]}</td>
                                            </tr>
                                        );
                                    }
                                )}
                        </tbody>
                    </table>
                    {/* <div className="aspects"></div> */}
                    <div className="evaluated-reviews">
                        {evaluation &&
                            evaluation.reviews &&
                            evaluation.reviews.map((review) => {
                                return (
                                    <div
                                        dangerouslySetInnerHTML={{
                                            __html: highlightReview(review),
                                        }}
                                    />
                                );
                            })}
                    </div>
                </div>
            </div>
            <div id="sidebar">
                <div className="file-picker-actions">
                    <Button
                        minimal={true}
                        intent={Intent.PRIMARY}
                        onClick={() => setImportDialogOpen(true)}
                    >
                        Import
                    </Button>
                    <Button
                        minimal={true}
                        intent={Intent.PRIMARY}
                        onClick={toggleReviewSelectMode}
                    >
                        Select..
                    </Button>
                    <Button
                        minimal={true}
                        intent={Intent.PRIMARY}
                        onClick={selectRandomReviews}
                    >
                        Select Randomly
                    </Button>
                    <Button
                        minimal={true}
                        intent={Intent.PRIMARY}
                        onClick={() => setSelectedReviews({})}
                    >
                        Clear Selection
                    </Button>
                    <div className="submit-reviews">
                        <Button
                            disabled={Object.keys(selectedReviews).length === 0}
                            intent={Intent.PRIMARY}
                            onClick={submitReviewsForAnalysis}
                        >
                            <span>
                                Submit
                                {Object.keys(selectedReviews).length > 0 && (
                                    <span>
                                        ({Object.keys(selectedReviews).length})
                                    </span>
                                )}
                            </span>
                        </Button>
                    </div>
                </div>
                <div className="entries">
                    {reviews.map(({ id, text }) => {
                        return (
                            <Card
                                className="card"
                                key={id}
                                elevation={Elevation.ONE}
                            >
                                {reviewSelectMode === true && (
                                    <div>
                                        <Checkbox
                                            checked={
                                                selectedReviews[id] ?? false
                                            }
                                            onChange={(event) =>
                                                selectReview({
                                                    id: id,
                                                    text: text,
                                                    checked:
                                                        event.target.checked,
                                                })
                                            }
                                        />
                                    </div>
                                )}
                                <div>{text}</div>
                            </Card>
                        );
                    })}
                </div>
            </div>
        </main>
    );
}

export default App;
