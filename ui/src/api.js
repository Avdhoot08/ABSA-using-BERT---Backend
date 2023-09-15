import axios from "axios";

const request = axios.create({
    baseURL: "http://localhost:5000/",
});

const submitReviews = async (reviews) => {
    const response = await request.post("/", reviews);
    return response.data;
};

export { submitReviews };
